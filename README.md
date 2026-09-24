# IM-Animation

[Paper](https://arxiv.org/abs/2602.07498) · [Project page](https://rabberk.github.io/IM-Animation/) · [Motion encoder weights](https://huggingface.co/Rbaerk/IM-Animation-Motion-Encoder)

**IM-Animation: An Implicit Motion Representation for Identity-decoupled Character Animation**

This release contains the final locally retained **TiTok-based motion encoder** implementation and an exported checkpoint. It provides frame-level motion tokens; it does not include the full animation generator or retargeting network. The project website and example videos remain available in this repository.

## Installation

```bash
git clone https://github.com/rabberk/IM-Animation.git
cd IM-Animation
pip install -r requirements.txt
```

The encoder was verified with PyTorch 2.7.1 on CPU using the exported BF16 weights.

## Download weights

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Rbaerk/IM-Animation-Motion-Encoder",
    filename="motion_encoder_latest.safetensors",
    local_dir=".",
)
```

The weight file is 607,017,208 bytes (approximately 579 MiB). Its SHA256 and provenance are in [`checkpoint_info.json`](checkpoint_info.json).

## Encode frames

Run from the repository directory:

```python
import torch
from motion_encoder import MotionEncoder

model = MotionEncoder.from_pretrained(device="cpu")  # or device="cuda"
frames = torch.rand(1, 3, 256, 256).to(
    device=next(model.parameters()).device,
    dtype=next(model.parameters()).dtype,
)
with torch.inference_mode():
    tokens, metrics = model.encode(frames)
print(tokens.shape)  # [1, 12, 1, 32]
```

`frames` must be RGB, normalized to `[0, 1]`, with shape `[N, 3, 256, 256]`.
The training preprocessing pads portrait frames horizontally to a square and resizes them to 256×256 with bilinear interpolation (`align_corners=False`). The original `HW_encoder_2` preprocessing class is included in `encoder_blocks.py`.

Each frame produces 32 tokens of 12 dimensions. The training integration flattens them into 384 dimensions per frame before retargeting. The encoder processes frames independently; temporal retargeting is outside this release.

## Architecture

- TiTokEncoder: 24 Transformer layers, hidden width 1024, 16 attention heads.
- Patch size 16; 32 learned latent tokens.
- 12-dimensional output projection and a 4096-entry vector-quantization codebook.
- Original `is_legacy=True` token reshape is retained for checkpoint compatibility.

Implementation: `motion_encoder.py` assembles the modules; `encoder_blocks.py` contains the original encoder and preprocessing; `quantizer.py` contains the original VQ implementation.

## Checkpoint provenance and verification

The selected checkpoint is `train_dit_5C_v6_part5/step-12200.safetensors`, dated 2025-11-28 UTC by file modification time. It is the newest checkpoint in the inspected local runs, rather than a claim of best quality or a verified paper-final checkpoint.

Training saved only trainable parameters. The selected checkpoint supplies 300 encoder/latent-token tensors. The frozen VQ codebook is restored from `train_motion_only_full_3C_20joint/step-3700.safetensors`, following the available training initialization code. This yields a complete 301-tensor encoding module. The historical run's frozen state has not been independently verified.

Validation checked tensor byte hashes against their source checkpoints, strict state-dict loading, and exact single-frame output agreement with the available original TiTok implementation. Full video-generation quality was not evaluated in this export.

## Acknowledgments and license

This encoder builds on [TiTok / 1d-tokenizer](https://github.com/bytedance/1d-tokenizer). Source attribution is preserved. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
