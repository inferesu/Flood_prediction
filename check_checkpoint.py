import torch

ckpt = torch.load("dane_anfis_model.pth", map_location="cpu")

print("Keys in checkpoint:")
for k, v in ckpt.items():
    if hasattr(v, 'shape'):
        print(f"  '{k}': tensor shape={v.shape}, has_nan={torch.isnan(v).any().item()}")
    else:
        print(f"  '{k}': {v}")