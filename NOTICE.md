<!-- SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc -->
<!--
SPDX-License-Identifier: MIT
-->

## Licenses for Special Components

This project includes third-party components originally licensed under terms different from the main project license.  
These components may **not** be used in commercial contexts due to upstream license restrictions.

Users are **responsible** for verifying compatibility with their use cases when adopting the following components.

| Component            | File(s)                                                                 | License           |
|----------------------|-------------------------------------------------------------------------|-------------------|
| `upfirdn2d`           | [mmcv/mmcv/ops/csrc/pytorch/cuda/upfirdn2d_kernel.cu](mmcv/mmcv/ops/csrc/pytorch/cuda/upfirdn2d_kernel.cu)                        | [NVIDIA License][nvidia-license] |
| `fused_leaky_relu`    | [mmcv/mmcv/ops/csrc/pytorch/cuda/fused_bias_leakyrelu_cuda.cu](mmcv/mmcv/ops/csrc/pytorch/cuda/fused_bias_leakyrelu_cuda.cu)              | [NVIDIA License][nvidia-license] |

These files originate from [NVIDIA StyleGAN2](https://github.com/NVlabs/stylegan2) and are governed by the following license:

> This work is made available under the NVIDIA Source Code License.  
> Redistribution and use are permitted for **non-commercial** purposes only.  
> See [NVIDIA License Terms][nvidia-license] for full details.

[nvidia-license]: https://nvlabs.github.io/stylegan2/license.html

