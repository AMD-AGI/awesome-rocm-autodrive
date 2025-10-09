# Copyright (C) OpenMMLab. All rights reserved.
#
# SPDX-License-Identifier: MIT
from .data_parallel import MLUDataParallel
from .distributed import MLUDistributedDataParallel

__all__ = ['MLUDataParallel', 'MLUDistributedDataParallel']
