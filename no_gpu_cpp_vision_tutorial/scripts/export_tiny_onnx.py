#!/usr/bin/env python3
"""Export a tiny ONNX model for 03_onnxruntime_cpu_infer.

Requires torch:

    python3 -m pip install torch onnx
    python3 scripts/export_tiny_onnx.py
"""
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "models" / "tiny_classifier.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        OUT.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
