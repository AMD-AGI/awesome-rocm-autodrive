# awesome-rocm-autodrive
A curated and practical collection of training examples, tools, and ROCm optimizations for key autonomous driving models. It aims to accelerate the deployment of deep learning models on AMD GPUs by providing out-of-the-box training support, Docker environments, and performance tuning utilities.

## 🔧 Supported Models (Initial Release)

| Model           | Type            | Repo Link                     | README for ROCm | Notes                              |
|------------------|------------------|--------------------------|---------------|-------------------------------------|
| ResNet-50        | Backbone         | https://github.com/amd-fuweiy/vision | [example/backbone](examples/backbone/readme.md)   |    |
| EfficientNet-B7  | Backbone         | https://github.com/amd-fuweiy/vision |    | Currently have performance issue with DWConv    |
| PointPillars     | Point Cloud      | https://github.com/Treemann/mmdetection3d | [examples/detection/pointpillars](examples/detection/pointpillars) | Need ROCm mmcv to get better performance          |
| MapTR            | Vector Prediction| https://github.com/aaab8b/MapTR | See modified Readme in git |  |
| FlashOcc         | Scene Occupancy  | https://github.com/mingjielu/FlashOCC | [examples/occupancy/FlashOCC](examples/occupancy/FlashOCC) |         |
| Sparse4D         | Sparse Detector  | https://github.com/binding7012/Sparse4D |[examples/detection/sparse4d](examples/detection/sparse4d) | |
| BEVFormer        | Multi-view       | https://github.com/jun-amd/BEVFormer | [examples/detection/BEVFormer](examples/detection/BEVFormer) |        |
| PETR             | 3D Detection | https://github.com/aaab8b/PETR | See modified Readme in git |  |
| QCNet            | Trajectory Prediction | https://github.com/aaab8b/QCNet | See modified Readme in git |  |
| SurroundOcc      | Scene Occupancy  | https://github.com/mingjielu/SurroundOcc | [examples/occupancy/FlashOCC](examples/occupancy/SurroundOcc) |         |
| UniAD            | Multi-model      | https://github.com/mingjielu/UniAD | [examples/Multi-model/UniAD](examples/Multi-model/UniAD) |         |



---

## 🚀 Quick Start

1. Clone the repo:

```bash
git clone https://github.com/YOUR_ORG/awesome-rocm-autodrive.git
cd awesome-rocm-autodrive



