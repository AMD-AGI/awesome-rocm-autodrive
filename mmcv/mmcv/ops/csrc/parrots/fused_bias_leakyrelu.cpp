// Modified from
// https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_bias_act.cpp

// Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
// Copyright Grant. Subject to the terms and conditions of this
//
// SPDX-License-Identifier: MIT

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor fused_bias_leakyrelu_op_impl(const torch::Tensor& input,
                                           const torch::Tensor& bias,
                                           const torch::Tensor& refer, int act,
                                           int grad, float alpha, float scale) {
  return DISPATCH_DEVICE_IMPL(fused_bias_leakyrelu_op_impl, input, bias, refer,
                              act, grad, alpha, scale);
}

torch::Tensor fused_bias_leakyrelu(const torch::Tensor& input,
                                   const torch::Tensor& bias,
                                   const torch::Tensor& refer, int act,
                                   int grad, float alpha, float scale) {
  return fused_bias_leakyrelu_op_impl(input, bias, refer, act, grad, alpha,
                                      scale);
}
