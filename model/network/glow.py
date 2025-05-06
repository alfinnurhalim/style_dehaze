import torch
from torch import nn
from torch.nn import functional as F

from model.layers import InvConv2dLU,InvConv2d,ZeroConv2d
from model.layers.activation_norm import SAN,ActNorm,AdaIN,AdaIN_SET
from model.layers.attention import SpatialAttention

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling,self).__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

        else:
            net_out = self.net(in_a)
            out_b = in_b - net_out

        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b + net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super(Flow,self).__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
            
        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        
        input = self.actnorm(input)
        input = self.invconv(input)
        #print('input: ',input.shape)
        if self.use_coupling:
            input = self.coupling(input)
        return input

    def reverse(self, input):
        if self.use_coupling:
            input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super(Block,self).__init__()

        # print('the in channel:',in_channel)
        squeeze_dim = in_channel * 4
        self.san = SAN(squeeze_dim,squeeze_dim)
        self.adain_set = AdaIN_SET()
        self.flows = nn.ModuleList()
        self.attn = SpatialAttention(squeeze_dim)

        self.last_attention = None

        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))
        
        # print('\n Block for',in_channel)
        # for name, param in self.attn.named_parameters():
        #     print(name, param.shape)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        
        for flow in self.flows:
            out = flow(out)

        return out

    def reverse(self, output, style, alpha=True):
        input = output

        # 1. Get style modulation parameters
        mean, std = self.san(input, style)

        # 2. Get attention map if enabled
        if alpha:
            attn_map = self.attn(input)  # [B, 1, H, W]
        else:
            attn_map = None

        # 3. AdaIN with optional spatial attention blending
        input = self.adain_set(input, mean, std, alpha=attn_map)

        if alpha and attn_map is not None:
          self.last_attention = attn_map.detach().cpu()

        # 4. Reverse through flow layers
        for flow in reversed(self.flows):
            input = flow.reverse(input)

        # 5. Un-squeeze back to spatial dimensions
        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        unsqueezed = unsqueezed.view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed

class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super(Glow,self).__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
            
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        
        
    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward_set(input)
        else:
            return self._reverse_set(input, style=style)

    def _forward_set(self, input):
        z = input
        
        for block in self.blocks:
            z = block(z)
        return z

    def _reverse_set(self, z, style):
        out = z
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out,style)
        return out



