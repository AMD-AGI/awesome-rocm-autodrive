## FlashOCC
1. Prepare nuScenes dataset as introduced in nuscenes_det.md and create the pkl for FlashOCC by running:
````
python tools/create_data_bevdet.py
````
2. For Occupancy Prediction task, download (only) the 'gts' from CVPR2023-3D-Occupancy-Prediction
3. rename customer ops in projects/mmdet3d_plugin/ops to solve "ninja: error: build.ninja:26: multiple rules generate" issue:
````
cd projects/mmdet3d_plugin/ops
mv bev_pool/src/bev_sum_pool_cuda.cu bev_pool/src/bev_sum_pool_hip_cuda.cu
mv bev_pool/src/bev_max_pool_cuda.cu bev_pool/src/bev_max_pool_hip_cuda.cu
mv bev_pool_v2/src/bev_pool_cuda.cu bev_pool_v2/src/bev_pool_hip_cuda.cu
mv nearest_assign/src/nearest_assign_cuda.cu nearest_assign/src/nearest_assign_hip_cuda.cu
````
4. Modify setup.py the cuda file name.
5. Modify projects/mmdet3d_plugin/core/evaluation/ray_metrics.py L13
````
#dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
dvr = load("dvr", sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=[])
````
6. Modify the cuda interface specially for torch2.7
````
lib/dvr/dvr.cu L371: sigma.type() -> sigma.scalar_type()
lib/dvr/dvr.cu L683: sigma.type() -> sigma.scalar_type()
lib/dvr/dvr.cu L736: points.type() -> points.scalar_type()
````
7. Modify get_ego_coor in projects/mmdet3d_plugin/models/necks/view_transformer.py L153 to speedup torch.bmm.
8. Training
````
python tools/train.py projects/configs/flashocc/flashocc-r50.py
````
