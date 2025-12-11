/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#ifndef MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH
#define MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

template <typename T>
__device__ __forceinline__ T dmcn_im2col_bilinear(const T *input, const int data_width,
                                  const int height, const int width, T h, T w) {
  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__device__ __forceinline__ T dmcn_get_gradient_weight(T argmax_h, T argmax_w, const int h,
                                      const int w, const int height,
                                      const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
__device__ __forceinline__ T dmcn_get_coordinate_weight(T argmax_h, T argmax_w,
                                        const int height, const int width,
                                        const T *im_data, const int data_width,
                                        const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T>
__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n, const T *data_im, const T *data_offset, const T *data_mask,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;

    const T *data_mask_ptr =
        data_mask + (b_col * deformable_group + deformable_group_index) *
                        kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im,
                                     w_im);
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

// Optimized col2im kernel - directly compute bilinear weights instead of 5x5 loop
template <typename T>
__global__ void modulated_deformable_col2im_gpu_kernel(
    const int n, const T *data_col, const T *data_offset, const T *data_mask,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    T *grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[index] * mask;
    
    // Early exit if out of bounds
    if (cur_inv_h_data <= -1 || cur_inv_w_data <= -1 || 
        cur_inv_h_data >= height || cur_inv_w_data >= width) {
      return;
    }
    
    const int cur_h = (int)floorf(cur_inv_h_data);
    const int cur_w = (int)floorf(cur_inv_w_data);
    
    // Bilinear interpolation weights
    const T lh = cur_inv_h_data - cur_h;
    const T lw = cur_inv_w_data - cur_w;
    const T hh = 1 - lh;
    const T hw = 1 - lw;
    
    T *grad_base = grad_im + (b * channels + c) * height * width;
    
    // Only 4 pixels can contribute (bilinear interpolation corners)
    if (cur_h >= 0 && cur_w >= 0) {
      atomicAdd(grad_base + cur_h * width + cur_w, hh * hw * cur_top_grad);
    }
    if (cur_h >= 0 && cur_w + 1 < width) {
      atomicAdd(grad_base + cur_h * width + cur_w + 1, hh * lw * cur_top_grad);
    }
    if (cur_h + 1 < height && cur_w >= 0) {
      atomicAdd(grad_base + (cur_h + 1) * width + cur_w, lh * hw * cur_top_grad);
    }
    if (cur_h + 1 < height && cur_w + 1 < width) {
      atomicAdd(grad_base + (cur_h + 1) * width + cur_w + 1, lh * lw * cur_top_grad);
    }
  }
}

// Optimized col2im_coord kernel - precompute interpolation weights and reuse
template <typename T>
__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n, const T *data_col, const T *data_im, const T *data_offset,
    const T *data_mask, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col, T *grad_offset, T *grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = 0, mval = 0;
    const int w = index % width_col;
    const int h = (index / width_col) % height_col;
    const int c = (index / width_col / height_col) % offset_channels;
    const int b = (index / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    
    const T *data_col_ptr = data_col + deformable_group_index *
                                           channel_per_deformable_group *
                                           batch_size * width_col * height_col;
    const T *data_im_ptr =
        data_im + (b * deformable_group + deformable_group_index) *
                      channel_per_deformable_group / kernel_h / kernel_w *
                      height * width;
    const T *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const T *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;
    const int bp_dir = offset_c % 2;
    const int kernel_idx = offset_c / 2;
    
    // Precompute the kernel position
    const int ki = kernel_idx / kernel_w;
    const int kj = kernel_idx % kernel_w;
    
    // Precompute offset indices - these are the same for all channel iterations
    const int hw_pos = h * width_col + w;
    const int data_offset_h_ptr = (2 * kernel_idx) * height_col * width_col + hw_pos;
    const int data_offset_w_ptr = (2 * kernel_idx + 1) * height_col * width_col + hw_pos;
    const int data_mask_hw_ptr = kernel_idx * height_col * width_col + hw_pos;
    
    // Load offset and mask ONCE - they are the same for all channel iterations
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    
    // Precompute the interpolation position
    const int w_in = w * stride_w - pad_w;
    const int h_in = h * stride_h - pad_h;
    const T inv_h = h_in + ki * dilation_h + offset_h;
    const T inv_w = w_in + kj * dilation_w + offset_w;
    
    // Precompute interpolation weights once
    const int h_low = floorf(inv_h);
    const int w_low = floorf(inv_w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;
    
    const T lh = inv_h - h_low;
    const T lw = inv_w - w_low;
    const T hh = 1 - lh;
    const T hw = 1 - lw;
    
    // Precompute validity flags
    const bool valid = (inv_h > -1 && inv_w > -1 && inv_h < height && inv_w < width);
    const bool v_ll = (h_low >= 0 && w_low >= 0);
    const bool v_lh = (h_low >= 0 && w_high <= width - 1);
    const bool v_hl = (h_high <= height - 1 && w_low >= 0);
    const bool v_hh = (h_high <= height - 1 && w_high <= width - 1);
    
    // Precompute indices
    const int idx_ll = h_low * width + w_low;
    const int idx_lh = h_low * width + w_high;
    const int idx_hl = h_high * width + w_low;
    const int idx_hh = h_high * width + w_high;
    
    // Precompute bilinear weights
    const T w_ll = hh * hw;
    const T w_lh = hh * lw;
    const T w_hl = lh * hw;
    const T w_hh = lh * lw;
    
    // Precompute coordinate weight coefficients
    T cw_ll, cw_lh, cw_hl, cw_hh;
    if (bp_dir == 0) {
      cw_ll = -hw; cw_lh = -lw; cw_hl = hw; cw_hh = lw;
    } else {
      cw_ll = -hh; cw_lh = hh; cw_hl = -lh; cw_hh = lh;
    }
    
    const int im_hw = height * width;
    const int col_stride = batch_size * height_col * width_col;
    int col_pos = (((kernel_idx * batch_size + b) * height_col) + h) * width_col + w;

    // Now iterate over channels with precomputed weights
    for (int col_c = kernel_idx; col_c < channel_per_deformable_group; col_c += col_step) {
      const T col_val = data_col_ptr[col_pos];
      const T *cur_data_im_ptr = data_im_ptr + cnt * im_hw;
      
      if (valid) {
        // Load image values
        T im_ll = 0, im_lh = 0, im_hl = 0, im_hh = 0;
        if (v_ll) im_ll = cur_data_im_ptr[idx_ll];
        if (v_lh) im_lh = cur_data_im_ptr[idx_lh];
        if (v_hl) im_hl = cur_data_im_ptr[idx_hl];
        if (v_hh) im_hh = cur_data_im_ptr[idx_hh];
        
        // Bilinear interpolation
        mval += col_val * (w_ll * im_ll + w_lh * im_lh + w_hl * im_hl + w_hh * im_hh);
        
        // Coordinate weight
        val += (cw_ll * im_ll + cw_lh * im_lh + cw_hl * im_hl + cw_hh * im_hh) * col_val * mask;
      }
      
      col_pos += col_step * col_stride;
      cnt += 1;
    }
    
    grad_offset[index] = val;
    if (bp_dir == 0) {
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w + kernel_idx) * height_col + h) * width_col + w] = mval;
    }
  }
}

#endif  // MODULATED_DEFORM_CONV_CUDA_KERNEL_CUH
