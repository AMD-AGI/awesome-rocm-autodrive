// Modified from
// https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.cpp

// Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
// Copyright Grant. Subject to the terms and conditions of this
//
// SPDX-License-Identifier: MIT

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor upfirdn2d_op_impl(const torch::Tensor& input,
                                const torch::Tensor& kernel, int up_x, int up_y,
                                int down_x, int down_y, int pad_x0, int pad_x1,
                                int pad_y0, int pad_y1) {
  return DISPATCH_DEVICE_IMPL(upfirdn2d_op_impl, input, kernel, up_x, up_y,
                              down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
}

torch::Tensor upfirdn2d(const torch::Tensor& input, const torch::Tensor& kernel,
                        int up_x, int up_y, int down_x, int down_y, int pad_x0,
                        int pad_x1, int pad_y0, int pad_y1) {
  return upfirdn2d_op_impl(input, kernel, up_x, up_y, down_x, down_y, pad_x0,
                           pad_x1, pad_y0, pad_y1);
}
