import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# Reference
# https://github.com/kjsman/stable-diffusion-pytorch

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_layer = nn.Identity()
        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residue)

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 8 * channels)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(-1, -2)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        xresidue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2).view((n, c, h, w))
        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x
        
class Switch(nn.Sequential):
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # input (3, 64, 64)
        self.encoders = nn.ModuleList([
            Switch(nn.Conv2d(in_channels, 160, kernel_size=3, padding=1)),                # input (160, 64, 64)
            Switch(ResidualBlock(160, 160), AttentionBlock(8, 20)),             # input (160, 64, 64)
            Switch(ResidualBlock(160, 160), AttentionBlock(8, 20)),             # input (160, 64, 64)
            Switch(nn.Conv2d(160, 160, kernel_size=3, stride=2, padding=1)),    # input (160, 32, 32)
            Switch(ResidualBlock(160, 320), AttentionBlock(8, 40)),             # input (320, 32, 32)
            Switch(ResidualBlock(320, 320), AttentionBlock(8, 40)),             # input (320, 32, 32)
            Switch(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),    # input (320, 32, 32)
            # Switch(ResidualBlock(640, 1280), AttentionBlock(8, 160)),           # input (1280, 32, 32)
            # Switch(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),          # input (1280, 32, 32)
            # Switch(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),  # input (1280, 16, 16)
            # Switch(ResidualBlock(1280, 1280)),
            # Switch(ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = Switch(ResidualBlock(320, 320), AttentionBlock(8, 40), ResidualBlock(320, 320)) # input (1280, 16, 16)

        self.decoders = nn.ModuleList([
            # Switch(ResidualBlock(2560, 1280)),
            # Switch(ResidualBlock(2560, 1280)),
            # Switch(ResidualBlock(2560, 1280), Upsample(1280)),                  # input (1280, 16, 16)
            # Switch(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),          # input (1280, 16, 16)
            # Switch(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),          # input (1280, 16, 16)
            Switch(ResidualBlock(640, 640), AttentionBlock(8, 80), Upsample(640)),# input (1280, 16, 16)
            Switch(ResidualBlock(960, 640), AttentionBlock(8, 80)),            # input (1280, 16, 16)
            Switch(ResidualBlock(960, 640), AttentionBlock(8, 80)),
            Switch(ResidualBlock(800, 640), AttentionBlock(8, 80), Upsample(640)),
            Switch(ResidualBlock(800, 320), AttentionBlock(8, 40)),
            Switch(ResidualBlock(480, 320), AttentionBlock(8, 40)),
            Switch(ResidualBlock(480, 320), AttentionBlock(8, 40)),
        ])
        self.time_embedding = TimeEmbedding(320)
        self.final_layer = FinalLayer(320, out_channels)

    def forward(self, x, time):
        time = self.time_embedding(time)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, time)
        x = self.final_layer(x)
        return x
    
    


if __name__ == '__main__':
    from torchinfo import summary
    # (3, 6) 110,884,646
    # (3, 3) 110,884,646
    test = UNet(3, 3) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.to(device)
    print(summary(test, input_size=[(1, 3, 64, 64), (1, 320)]))