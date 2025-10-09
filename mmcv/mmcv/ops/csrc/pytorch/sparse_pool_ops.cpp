// Copyright 2019 Yan Yan
//
// SPDX-License-Identifier: MIT

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor indice_maxpool_forward_impl(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          int64_t numAct) {
  return DISPATCH_DEVICE_IMPL(indice_maxpool_forward_impl, features,
                              indicePairs, indiceNum, numAct);
}

torch::Tensor indice_maxpool_forward(torch::Tensor features,
                                     torch::Tensor indicePairs,
                                     torch::Tensor indiceNum, int64_t numAct) {
  return indice_maxpool_forward_impl(features, indicePairs, indiceNum, numAct);
}

torch::Tensor indice_maxpool_backward_impl(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum) {
  return DISPATCH_DEVICE_IMPL(indice_maxpool_backward_impl, features,
                              outFeatures, outGrad, indicePairs, indiceNum);
}

torch::Tensor indice_maxpool_backward(torch::Tensor features,
                                      torch::Tensor outFeatures,
                                      torch::Tensor outGrad,
                                      torch::Tensor indicePairs,
                                      torch::Tensor indiceNum) {
  return indice_maxpool_backward_impl(features, outFeatures, outGrad,
                                      indicePairs, indiceNum);
}
