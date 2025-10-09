// Copyright (C) 2022 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: MIT

#ifndef PYTORCH_NPU_HELPER_HPP_
#define PYTORCH_NPU_HELPER_HPP_

#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#define NPU_NAME_SPACE at_npu::native

#define REGISTER_NPU_IMPL(key, value) REGISTER_DEVICE_IMPL(key, XLA, value)

#define CHECK_NPU(x) \
  TORCH_CHECK(x.device().type() == at::kXLA, #x " must be a NPU tensor")

#endif  // PYTORCH_NPU_HELPER_HPP_
