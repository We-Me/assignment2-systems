import torch
from torch import nn
from torch.nn import functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("fc1 output:", x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("LayerNorm output:", x.dtype)
        return self.fc2(x)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires a CUDA GPU.")

    model = ToyModel(in_features=8, out_features=3).to(device="cuda", dtype=torch.float32)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 3, (4,), device="cuda")

    for label, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
        print(f"\n=== {label} autocast ===")
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            for name, parameter in model.named_parameters():
                print(f"Parameter {name}:", parameter.dtype)
            logits = model(x)
            print("Logits:", logits.dtype)
            loss = F.cross_entropy(logits, targets)
            print("Loss (cross_entropy):", loss.dtype)

        loss.backward()
        for name, parameter in model.named_parameters():
            print(f"Gradient {name}:", parameter.grad.dtype)
