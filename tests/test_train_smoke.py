import torch
from models import SimpleCNN

def test_forward_pass():
    model = SimpleCNN()
    x = torch.randn(4, 3, 32, 32)  # 4 RGB images
    y = model(x)
    assert y.shape == (4, 10)
