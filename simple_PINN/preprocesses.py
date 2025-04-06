import numpy as np
import random
import torch
from simple_PINN.settings import (
    N_INITIAL, 
    N_BOUNDARY, 
    N_REGION,
)

def torch_fix_seed(seed=42):
    """ 乱数シードを固定
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
