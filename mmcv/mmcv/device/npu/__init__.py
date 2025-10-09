# Copyright Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) OpenMMLab. All rights reserved.
#
# SPDX-License-Identifier: MIT
from .data_parallel import NPUDataParallel
from .distributed import NPUDistributedDataParallel

__all__ = ['NPUDataParallel', 'NPUDistributedDataParallel']
