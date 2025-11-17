"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover',
    'high_prec_timing', 'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio',
    'stream_elas', 'slope_fdc', 'baseflow_index', 'hfd_mean', 'q5', 'q95',
    'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq',
    'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

# NLDAS mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    'input_means': np.array([3.015, 357.68, 10.864, 10.864, 1055.533]),
    'input_stds': np.array([7.573, 129.878, 10.932, 10.932, 705.998]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672]),
}

camels_root = r"F:\CAMEL_Far\CAMELS_US"

def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table."""
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(
    db_path: str,
    basins: List,
    drop_lat_lon: bool = True,
    keep_features: List = None
) -> pd.DataFrame:
    """Load attributes from database file into DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics."""
    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")
    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale normalized features back to original scale."""
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")
    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples."""
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files."""
    forcing_path = camels_root / 'basin_mean_forcing' / 'nldas_extended'

    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]

    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """Load USGS discharge data and normalize to mm/day."""
    discharge_path = camels_root / 'usgs_streamflow'

    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]

    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)

    dates = df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs
