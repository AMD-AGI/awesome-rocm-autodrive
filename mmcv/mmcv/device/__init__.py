# Copyright (C) OpenMMLab. All rights reserved.
#
# SPDX-License-Identifier: MIT
from . import ipu, mlu, mps, npu
from .scatter_gather import scatter, scatter_kwargs
from .utils import get_device

__all__ = [
    'npu', 'mlu', 'ipu', 'mps', 'get_device', 'scatter', 'scatter_kwargs'
]
