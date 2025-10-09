// Copyright (C) 2022 Cambricon.
//
// SPDX-License-Identifier: MIT
#ifndef ROI_ALIGN_ROTATED_UTILS_HPP_
#define ROI_ALIGN_ROTATED_UTILS_HPP_

struct RoiAlignRotatedParams {
  int pooled_height;
  int pooled_width;
  int sample_ratio;
  float spatial_scale;
  bool aligned;
  bool clockwise;
};

#endif  // ROI_ALIGN_ROTATED_UTILS_HPP_
