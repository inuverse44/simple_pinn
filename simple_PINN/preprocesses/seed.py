import numpy as np
import random
import torch

def torch_fix_seed(seed=42):
    """
    Fix random seeds for reproducibility across NumPy, random, and PyTorch.

    This ensures deterministic behavior in model training and data generation.

    Parameters:
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
