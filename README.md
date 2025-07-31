# awesome-rocm-autodrive
**awesome-rocm-autodrive-training** is a collection of autonomous driving model training examples adapted and optimized for AMD GPUs using the ROCm platform.  
This project provides an out-of-the-box (OOTB) training experience through ROCm-compatible MMCV, prebuilt Docker environments, and a wide range of real-world autonomous driving model examples.

---

## 🚀 Key Highlights

- ✅ **Out-of-the-box training** experience on AMD ROCm GPUs
- 🛠️ **Patched MMCV** with full ROCm compatibility and performance fixes
- 🐳 **Ready-to-use Docker** environment and image
- 🧩 **Diverse training examples** across major autonomous driving tasks
- ⚡ ROCm-specific **performance optimizations** in selected models

---

## 📁 Repository Structure

```text
awesome-rocm-autodrive-training/
├── docker/          # Dockerfile and prebuilt ROCm training environment
├── examples/        # Training examples for various AD tasks
│   ├── backbone/    # Backbone networks (e.g., ResNet50, EfficientNet)
│   ├── 3d_detection/   # 2D/3D object detection (e.g., PointPillars)
│   ├── prediction/       # Prediction models (e.g., QCNet)
│   ├── bev/         # BEV perception models (e.g., BEVFormer)
│   ├── mapping/         # HD map construction models (e.g., MapTR)
│   ├── occupancy/   # Occupancy prediction (e.g., FlashOcc, SurroundOcc)
│   ├── end2end/         # End-to-end driving pipelines (e.g., UniAD)
├── mmcv/            # ROCm-adapted mmcv source code
├── tools/           # Utility scripts for benchmarking/tuning
└── README.md
```
---

## 🔧 Supported Models (Initial Release)

| Model           | Type            | Repo Link                     | README for ROCm | Notes                              |
|------------------|------------------|--------------------------|---------------|-------------------------------------|
| ResNet-50        | Backbone         | https://github.com/amd-fuweiy/vision | [example/backbone](examples/backbone/readme.md)   |    |
| EfficientNet-B7  | Backbone         | https://github.com/amd-fuweiy/vision |    | Currently have performance issue with DWConv    |
| PointPillars     | Point Cloud      | https://github.com/Treemann/mmdetection3d | [examples/detection/pointpillars](examples/3d_detection/pointpillars) | Need ROCm mmcv to get better performance          |
| MapTR            | Vector Prediction| https://github.com/aaab8b/MapTR | See modified Readme in git |  |
| FlashOcc         | Scene Occupancy  | https://github.com/mingjielu/FlashOCC | [examples/occupancy/FlashOCC](examples/occupancy/FlashOCC) |         |
| Sparse4D         | Sparse Detector  | https://github.com/binding7012/Sparse4D |[examples/detection/sparse4d](examples/3d_detection/sparse4d) | |
| BEVFormer        | Multi-view       | https://github.com/jun-amd/BEVFormer | [examples/detection/BEVFormer](examples/BEV/BEVFormer) |        |
| PETR             | 3D Detection | https://github.com/aaab8b/PETR | See modified Readme in git |  |
| QCNet            | Trajectory Prediction | https://github.com/aaab8b/QCNet | See modified Readme in git |  |
| SurroundOcc      | Scene Occupancy  | https://github.com/mingjielu/SurroundOcc | [examples/occupancy/FlashOCC](examples/occupancy/SurroundOcc) |         |
| UniAD            | Multi-model      | https://github.com/mingjielu/UniAD | [examples/Multi-model/UniAD](examples/end2end/UniAD) |         |


---

## 🚀 Quick Start

### 1. Clone the repo:

```bash
git clone https://github.com/AMD-AIG-AIMA/awesome-rocm-autodrive.git
cd awesome-rocm-autodrive
```

### 2. Build Docker Image
```bash
cd docker
docker build -t rocm-autodrive .
```

or directly pull the docker image provided by AMD:
```bash
docker pull rocm/pytorch-training:autodrive
```

### 3. Luanch Docker Container
```bash
docker run --rm -it --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $PWD:/workspace \
  rocm-autodrive
```

### 4. Run an Example
git clone the repo link provided in README, repare dataset, and run as corresponding readme.



## 🔮 What's Next

We are actively expanding the coverage of this project. Upcoming efforts include:

- ✅ **More Models**: Support for popular/SOTA models such as DeepAccident, and end-to-end transformer-based planners.
- 🔍 **Performance Benchmarking**: Add ROCm vs CUDA training benchmark results across all models.
- 🧠 **FP8 & Mixed Precision Training**: Explore FP8/BF16 optimization on MI300X GPUs.
- 🤝 **Upstream Collaboration**: Work with upstream repos to upstream ROCm compatibility patches and improvements.

Have ideas or requests? [Open an issue](https://github.com/AMD-AIG-AIMA/awesome-rocm-autodrive/issues) or start a discussion!

---

## 🤝 Contributing

We welcome contributions from the community!

To contribute:

1. Fork this repository.
2. Clone your fork and create a feature branch.
3. Make your changes and test them.
4. Submit a pull request with a clear description.

When adding a new model:

- ✅ Include training scripts and a minimal README.
- ⚙️ Note any ROCm-specific patches or performance tips.
- 🧪 Validate functionality on at least one ROCm GPU (MI300X, MI325X, or MI355X).
- 📢 Consider submitting upstream patches for long-term compatibility.
 
---

## 📝 License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this codebase for commercial and non-commercial purposes, under the terms of the license.

See the full license text here: [LICENSE](./LICENSE)


