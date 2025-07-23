## SurroundOcc
1. Insert torch.multiprocessing.set_start_method('fork') before the main() call at line 256 of tools/train.py to fix TypeError: cannot pickle 'dict_keys' object.
