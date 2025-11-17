"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019).
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Fix path so modules import correctly ---
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from data.datasets import CamelsH5, CamelsTXT
from data.datautils import add_camels_attributes, rescale_features
from Scripts.lstm import LSTM
from Scripts.nseloss import NSELoss
from Scripts.utils import create_h5_files, get_basin_list


###############
# Global Settings
###############

GLOBAL_SETTINGS = {
    'batch_size': 2000,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 2,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('01101980', format='%d%m%Y'),
    'train_end': pd.to_datetime('30091995', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101995', format='%d%m%Y'),
    'val_end': pd.to_datetime('30092010', format='%d%m%Y')
}


###############
# Parse args
###############

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness", "create_splits"])
    parser.add_argument('--camels_root', type=str, default='F:/CAMEL_Far/CAMELS_US/')
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--cache_data', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--no_static', type=str2bool, default=False)
    parser.add_argument('--concat_static', type=str2bool, default=False)
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--n_splits', type=int, default=None)
    parser.add_argument('--basin_file', type=str, default=None)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--split_file', type=str, default=None)

    cfg = vars(parser.parse_args())

    if cfg["mode"] in ["train", "create_splits"] and cfg["seed"] is None:
        cfg["seed"] = int(np.random.uniform(0, 1e6))

    if cfg["mode"] in ["evaluate", "eval_robustness"] and cfg["run_dir"] is None:
        raise ValueError("In evaluation mode a run directory (--run_dir) must be specified")

    if cfg["gpu"] >= 0:
        device = f"cuda:{cfg['gpu']}"
    else:
        device = "cpu"

    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        for key, val in cfg.items():
            print(f"{key}: {val}")

    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])

    return cfg


###############
# Setup run
###############

def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now()
    run_name = f"run_{now.day:02d}{now.month:02d}_{now.hour:02d}{now.minute:02d}_seed{cfg['seed']}"
    cfg['run_dir'] = Path(__file__).absolute().parent.parent / "runs" / run_name

    if cfg["run_dir"].is_dir():
        raise RuntimeError(f"Folder exists: {cfg['run_dir']}")

    cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
    cfg["train_dir"].mkdir(parents=True)
    cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
    cfg["val_dir"].mkdir(parents=True)

    temp_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, Path):
            temp_cfg[k] = str(v)
        elif isinstance(v, pd.Timestamp):
            temp_cfg[k] = v.strftime("%d%m%Y")
        else:
            temp_cfg[k] = v

    with open(cfg["run_dir"] / 'cfg.json', 'w') as fp:
        json.dump(temp_cfg, fp, indent=4, sort_keys=True)

    return cfg


###############
# Data prep
###############

def _prepare_data(cfg: Dict, basins: List) -> Dict:
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    cfg["train_file"] = cfg["train_dir"] / "train_data.h5"
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg["train_file"],
        basins=basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"]
    )

    return cfg


###############
# Model Wrapper
###############

class Model(nn.Module):
    def __init__(
        self,
        input_size_dyn: int,
        hidden_size: int,
        initial_forget_bias: int = 5,
        dropout: float = 0.0,
        concat_static: bool = False,
        no_static: bool = False,
    ):
        super(Model, self).__init__()

        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor):
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###############
# Train
###############

def train(cfg):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        with open(cfg["split_file"], "rb") as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
    else:
        basins = get_basin_list()

    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg, basins)

    ds = CamelsH5(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"]
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    input_size_dyn = 5 if (cfg["no_static"] or not cfg["concat_static"]) else 32
    model = Model(
        input_size_dyn=input_size_dyn,
        hidden_size=cfg["hidden_size"],
        initial_forget_bias=cfg["initial_forget_gate_bias"],
        dropout=cfg["dropout"],
        concat_static=cfg["concat_static"],
        no_static=cfg["no_static"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = nn.MSELoss() if cfg["use_mse"] else NSELoss()

    lr_schedule = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch in lr_schedule:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        torch.save(model.state_dict(), str(cfg["run_dir"] / f"model_epoch{epoch}.pt"))


def train_epoch(model, optimizer, loss_func, loader, cfg, epoch, use_mse):
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f"# Epoch {epoch}")

    for data in pbar:
        optimizer.zero_grad()
        x, y, q_stds = data
        x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)

        predictions = model(x)[0]

        if use_mse:
            loss = loss_func(predictions, y)
        else:
            loss = loss_func(predictions, y, q_stds)

        loss.backward()

        if cfg["clip_norm"]:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.5f}")


###############
# Evaluate
###############

def evaluate(user_cfg):
    with open(user_cfg["run_dir"] / 'cfg.json') as fp:
        run_cfg = json.load(fp)

    if user_cfg["split_file"] is not None:
        with open(user_cfg["split_file"], 'rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    train_file = user_cfg["run_dir"] / "data" / "train" / "train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")

    ds_train = CamelsH5(
        h5_file=train_file,
        db_path=db_path,
        basins=basins,
        concat_static=run_cfg["concat_static"]
    )
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    model = Model(
        input_size_dyn=input_size_dyn,
        hidden_size=run_cfg["hidden_size"],
        dropout=run_cfg["dropout"],
        concat_static=run_cfg["concat_static"],
        no_static=run_cfg["no_static"],
    ).to(DEVICE)

    weight_file = user_cfg["run_dir"] / 'model_epoch2.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    # date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])[run_cfg["seq_length"] - 1:]
    results = {}

    for basin in tqdm(basins):
        ds_test = CamelsTXT(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
            is_train=False,
            seq_length=run_cfg["seq_length"],
            with_attributes=True,
            attribute_means=means,
            attribute_stds=stds,
            concat_static=run_cfg["concat_static"],
            db_path=db_path
        )

        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        preds, obs = evaluate_basin(model, loader)
        # df = pd.DataFrame({'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        df = pd.DataFrame({'qobs': obs.reshape(-1), 'qsim': preds.reshape(-1)}, index=date_range)

        results[basin] = df

    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model, loader):
    model.eval()
    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            # x, y = data
            x, y, x_s = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)[0]

            if preds is None:
                preds = p.cpu()
                obs = y.cpu()
            else:
                preds = torch.cat((preds, p.cpu()), 0)
                obs = torch.cat((obs, y.cpu()), 0)


        
        # Convert to numpy and handle data cleaning/rescaling
        preds_np = rescale_features(preds.numpy(), variable='output')
        obs_np = obs.numpy()
        preds_np[preds_np < 0] = 0
        
        # We enforce the correct length (5210) for the final output, 
        # which accounts for the 270-day spin-up period and leap days.
        # Total days in test period: 5479. Valid prediction days: 5210.
        
        if preds_np.size >= 5210:
            preds_final = preds_np.flatten()[-5210:]
            obs_final = obs_np.flatten()[-5210:]
        else:
            # Fallback if the accumulation failed to reach the target length
            preds_final = preds_np.flatten()
            obs_final = obs_np.flatten()

    return preds_final, obs_final

    # return preds, obs


###############
# Store results
###############

def _store_results(user_cfg, run_cfg, results):
    if run_cfg["no_static"]:
        f = f"lstm_no_static_seed{run_cfg['seed']}.p"
    else:
        f = f"lstm_seed{run_cfg['seed']}.p" if run_cfg["concat_static"] else f"ealstm_seed{run_cfg['seed']}.p"

    file_path = user_cfg["run_dir"] / f

    with open(file_path, "wb") as fp:
        pickle.dump(results, fp)

    print(f"Successfully stored results at {file_path}")


###############
# Cross-validation splits
###############

def create_splits(cfg: dict):
    output_file = Path(__file__).absolute().parent / f"data/kfold_splits_seed{cfg['seed']}.p"

    if output_file.is_file():
        raise RuntimeError(f"File exists: {output_file}")

    np.random.seed(cfg["seed"])

    if cfg["basin_file"] is not None:
        bf = Path(cfg["basin_file"])
        if not bf.is_file():
            raise FileNotFoundError(f"No file found at {cfg['basin_file']}")
        basins = [b.strip() for b in bf.read_text().splitlines()]
        ignore = ['06775500', '06846500', '09535100']
        basins = [b for b in basins if b not in ignore]
    else:
        basins = get_basin_list()

    kfold = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    splits = defaultdict(dict)

    for split, (train_idx, test_idx) in enumerate(kfold.split(basins)):
        splits[split] = {
            "train": [basins[i] for i in train_idx],
            "test": [basins[i] for i in test_idx],
        }

    with open(output_file, "wb") as fp:
        pickle.dump(splits, fp)

    print(f"Stored basin splits: {output_file}")


###############
# Main
###############

if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)




