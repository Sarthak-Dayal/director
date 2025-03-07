import torch
from torch import nn as nn

class DummyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.ones(1))
        self.b = None

    def forward(self, x, y):
        if self.b is None:
            b = nn.Parameter(torch.ones(1))
            setattr(self, "b", b)

        return self.a * x + self.b * y


if __name__ == '__main__':
    torch.set_default_device("cuda:0")
    model = DummyModel()
    model = model.cuda()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  {model.__class__.__name__}.{name} has no gradient!")
        else:
            print(f"  {model.__class__.__name__}.{name} has gradient!")
    out = model.forward(torch.ones(1), torch.ones(1))
    print(out)
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  {model.__class__.__name__}.{name} has no gradient!")
        else:
            print(f"  {model.__class__.__name__}.{name} has gradient!")
    out.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  {model.__class__.__name__}.{name} has no gradient!")
        else:
            print(f"  {model.__class__.__name__}.{name} has gradient!")