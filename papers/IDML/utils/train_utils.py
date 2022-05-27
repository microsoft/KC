import random
import numpy as np
import torch
import torch.nn.functional as F

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return

def euclidean_dist(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.pow(x - y, 2).sum(2)


def cosine_similarity(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    sim = torch.matmul(x, y.transpose(1, 0))
    return sim