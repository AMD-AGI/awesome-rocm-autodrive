// Copyright (C) Facebook, Inc. and its affiliates. All Rights Reserved
//
// SPDX-License-Identifier: MIT
#include "box_iou_rotated_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned) {
  using scalar_t = float;
  AT_ASSERTM(boxes1.is_cuda(), "boxes1 must be a CUDA tensor");
  AT_ASSERTM(boxes2.is_cuda(), "boxes2 must be a CUDA tensor");

  int output_size = ious.numel();
  int num_boxes1 = boxes1.size(0);
  int num_boxes2 = boxes2.size(0);

  at::cuda::CUDAGuard device_guard(boxes1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  box_iou_rotated_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          num_boxes1, num_boxes2, boxes1.data_ptr<scalar_t>(),
          boxes2.data_ptr<scalar_t>(), (scalar_t*)ious.data_ptr<scalar_t>(),
          mode_flag, aligned);
  AT_CUDA_CHECK(cudaGetLastError());
}
