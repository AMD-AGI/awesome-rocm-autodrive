## PointPillars
1. Dataset Preparation
- Check and follow the instruction in `scripts/prepare_nuscenes.sh` to set related variables properly and run the script to prepare the nuScenes dataset.
- It will take a long time (a few hours) to download and process the dataset.
```bash
bash scripts/prepare_nuscenes.sh
```
2. Model Training & Evaluation
- Set the config file, working directory, GPU numbers etc. in `scripts/run_pointpillars.sh`, then run it.
- Note: our optimization for voxelization has been integrated into docker to accelerate the performance of training and inference.
```bash
bash scripts/run_pointpillars.sh
```
