
import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, mid_channels=64, kernel_size=7):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attn(x)



'''
Yes — it’s intentionally shallow.
And here's why:

✅ Why It's Typically Shallow
This comes from CBAM (Convolutional Block Attention Module) [Woo et al., ECCV 2018].

They found that a single conv layer is enough to capture coarse attention patterns without introducing instability or overfitting.

Shallow attention is:

Fast and light

Easy to integrate into deep architectures

Surprisingly effective for many tasks


'''