# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    import MultiScaleDeformableAttention as MSDA
    print("Success")
except Exception as e:
    print("Failed to import 'MultiScaleDeformableAttention' in test_import_1.py")
    print(e)
