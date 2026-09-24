# Extracted for IM-Animation: encoder and preprocessing only; original forward logic retained.
"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from collections import OrderedDict
from einops import rearrange

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.width_size = config.dataset.preprocessing.width_size
        self.height_size = config.dataset.preprocessing.height_size
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size_w = self.width_size // self.patch_size
        self.grid_size_h = self.height_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size,padding = (4,2), bias=True)

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size_h*self.grid_size_w + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]


        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            # x = self.transformer[i](x)
            #with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.transformer[i]), x,use_reentrant=False)

        x = x.permute(1, 0, 2)  # LND -> NLD

        latent_tokens = x[:, 1+self.grid_size_h*self.grid_size_w:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens

class HW_encoder_2(nn.Module):
    def __init__(self, in_channels):
        super(HW_encoder_2, self).__init__()
    #     self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    #     self.conv1 = nn.Conv2d(in_channels*4, in_channels, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(in_channels*4 , in_channels, kernel_size=3, padding=1)
    #     self.conv3 = nn.Conv2d(in_channels*4 , in_channels , kernel_size=3, padding=1)
    #     # self.conv4 = nn.Conv2d(in_channels , in_channels//4 , kernel_size=3, padding=1)

    # def pixel_shuffle(self, x, scale_factor=0.5):
    #     n, c, h, w = x.size()
    #     new_h = int(h * scale_factor)
    #     new_w = int(w * scale_factor)
    #     x = x.view(n, int(c / (scale_factor ** 2)), new_h, new_w)

        # return x
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        # Step 1: Pad the width from 480 to 832
        padding_width = (H - W) // 2
        x = F.pad(x, (padding_width, padding_width, 0, 0))  # Pad width only

        # Step 2: Resize to target width 256
        target_size = (256, 256)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # Rearrange back to original shape
        x = rearrange(x, "(b f) c h w -> b c f h w", f=T)

        return x
