## BEVFormer

1. Install dependency
````
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0
````
2. Prepare data 
````
# prepare data before install detectron2, otherwise cause path error.
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
````
3. Install Detectron2
````
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
````
4. Insert `torch.multiprocessing.set_start_method('fork')` before the `main()` call at **line 271** of `tools/train.py` to fix **TypeError: cannot pickle 'dict_keys' object**. 
5. Run the scripts:
````
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_tiny.py 8
````
