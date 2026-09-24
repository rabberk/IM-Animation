"""Encoder-only assembly of the original TiTok motion encoding path.

TiTokEncoder, VectorQuantizer, HW_encoder_2 retain the original implementations.
Input to encode: preprocessed frames [N, 3, 256, 256], RGB in [0, 1].
"""
from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from safetensors.torch import load_file
from encoder_blocks import TiTokEncoder, HW_encoder_2
from quantizer import VectorQuantizer

class MotionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TiTokEncoder(config)
        vq = config.model.vq_model
        self.latent_tokens = nn.Parameter(torch.empty(vq.num_latent_tokens, self.encoder.width))
        self.quantize = VectorQuantizer(codebook_size=vq.codebook_size,
            token_size=vq.token_size, commitment_cost=vq.commitment_cost,
            use_l2_norm=vq.use_l2_norm)

    def encode(self, frames):
        z = self.encoder(pixel_values=frames, latent_tokens=self.latent_tokens)
        return self.quantize(z)

    def forward(self, frames):
        return self.encode(frames)

    @classmethod
    def from_pretrained(cls, directory=None, device='cpu'):
        directory = Path(directory or Path(__file__).parent)
        config = OmegaConf.load(directory / 'config.yaml')
        with torch.device('meta'):
            model = cls(config)
        model.load_state_dict(load_file(str(directory / 'motion_encoder_latest.safetensors')), strict=True, assign=True)
        return model.to(device).eval()
