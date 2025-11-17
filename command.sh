# Training command:
for ($i =1; $i -le 12; $i++) {Write-Host "--- Starting training for Seed $i ---" python src/main.py train --seed $i --no_static True  Write-Host "--- Finished training for Seed $i ---" }

# evaluation command:
$basePath = "C:\Users\hdagne1\Box\NRT_Project_2025Fall\Habtamu\HydroAuditToolFrameowrk\runs"; Get-ChildItem -Path $basePath -Directory | Where-Object { $_.Name -like "run_*" } | ForEach-Object { $run_path = $_.FullName; Write-Host "--- Starting evaluation for $($_.Name) ---"; python src/main.py evaluate --run_dir $run_path --gpu -1; Write-Host "--- Finished evaluation for $($_.Name) ---" }