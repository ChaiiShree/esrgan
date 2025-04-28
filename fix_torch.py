import os
import torch

# Fix the torch classes path issue
torch.classes.__path__ = []
