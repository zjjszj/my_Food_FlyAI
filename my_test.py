import numpy as np
import torch

a=torch.tensor([[1,10,8],[1,4,2],[5,3,1]])
pre=a.max(1)[1]
print(pre)