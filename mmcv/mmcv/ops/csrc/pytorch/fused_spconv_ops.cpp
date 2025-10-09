// Copyright 2019 Yan Yan
//
// SPDX-License-Identifier: MIT

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor fused_indice_conv_batchnorm_forward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return DISPATCH_DEVICE_IMPL(fused_indice_conv_batchnorm_forward_impl,
                              features, filters, bias, indicePairs, indiceNum,
                              numActOut, _inverse, _subM);
}

torch::Tensor fused_indice_conv_batchnorm_forward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t numActOut,
    int64_t _inverse, int64_t _subM) {
  return fused_indice_conv_batchnorm_forward_impl(features, filters, bias,
                                                  indicePairs, indiceNum,
                                                  numActOut, _inverse, _subM);
}
