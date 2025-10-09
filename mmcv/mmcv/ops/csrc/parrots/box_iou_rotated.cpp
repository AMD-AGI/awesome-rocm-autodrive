// Copyright (C) Facebook, Inc. and its affiliates. All Rights Reserved
//
// SPDX-License-Identifier: MIT
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void box_iou_rotated_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned) {
  DISPATCH_DEVICE_IMPL(box_iou_rotated_impl, boxes1, boxes2, ious, mode_flag,
                       aligned);
}

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                     const int mode_flag, const bool aligned) {
  box_iou_rotated_impl(boxes1, boxes2, ious, mode_flag, aligned);
}
