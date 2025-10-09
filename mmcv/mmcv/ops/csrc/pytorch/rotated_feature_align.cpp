// Copyright (C) OpenMMLab. All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void rotated_feature_align_forward_impl(const Tensor features,
                                        const Tensor best_bboxes,
                                        const float spatial_scale,
                                        const int points, Tensor output) {
  DISPATCH_DEVICE_IMPL(rotated_feature_align_forward_impl, features,
                       best_bboxes, spatial_scale, points, output);
}

void rotated_feature_align_backward_impl(const Tensor top_grad,
                                         const Tensor best_bboxes,
                                         const float spatial_scale,
                                         const int points, Tensor bottom_grad) {
  DISPATCH_DEVICE_IMPL(rotated_feature_align_backward_impl, top_grad,
                       best_bboxes, spatial_scale, points, bottom_grad);
}

void rotated_feature_align_forward(const Tensor features,
                                   const Tensor best_bboxes, Tensor output,
                                   const float spatial_scale,
                                   const int points) {
  rotated_feature_align_forward_impl(features, best_bboxes, spatial_scale,
                                     points, output);
}

void rotated_feature_align_backward(const Tensor top_grad,
                                    const Tensor best_bboxes,
                                    Tensor bottom_grad,
                                    const float spatial_scale,
                                    const int points) {
  rotated_feature_align_backward_impl(top_grad, best_bboxes, spatial_scale,
                                      points, bottom_grad);
}
