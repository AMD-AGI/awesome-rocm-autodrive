<!--
SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc

SPDX-License-Identifier: MIT
-->
## UniAD
1. Insert torch.multiprocessing.set_start_method('fork') before the main() call at line 260 of tools/train.py to fix TypeError: cannot pickle 'dict_keys' object.
