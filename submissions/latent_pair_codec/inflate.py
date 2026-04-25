#!/usr/bin/env python3
import io
import sys
from pathlib import Path

import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import camera_size

PAIR_LEN = 2
LATENT_C = 2
LATENT_H = 12
LATENT_W = 16


def choose_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


class SepConvGNAct(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, depth_mult: int = 2):
    super().__init__()
    mid = in_ch * depth_mult
    self.dw = nn.Conv2d(in_ch, mid, 3, padding=1, groups=in_ch, bias=False)
    self.pw = nn.Conv2d(mid, out_ch, 1)
    self.norm = nn.GroupNorm(4 if out_ch >= 4 else 1, out_ch)
    self.act = nn.SiLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.act(self.norm(self.pw(self.dw(x))))


class SepResBlock(nn.Module):
  def __init__(self, ch: int):
    super().__init__()
    self.block1 = SepConvGNAct(ch, ch)
    self.block2 = nn.Sequential(
      nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
      nn.Conv2d(ch, ch, 1),
      nn.GroupNorm(4 if ch >= 4 else 1, ch),
    )
    self.act = nn.SiLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.act(x + self.block2(self.block1(x)))


def make_coord_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
  ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
  xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
  yy, xx = torch.meshgrid(ys, xs, indexing="ij")
  grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
  return grid.unsqueeze(0).expand(batch, -1, -1, -1)


class PairLatentDecoder(nn.Module):
  def __init__(self, latent_channels: int = LATENT_C):
    super().__init__()
    in_ch = latent_channels + 2
    self.stem = SepConvGNAct(in_ch, 32)
    self.block1 = SepResBlock(32)
    self.up1 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
      SepConvGNAct(32, 40),
      SepResBlock(40),
    )
    self.up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
      SepConvGNAct(40, 48),
      SepResBlock(48),
    )
    self.up3 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
      SepConvGNAct(48, 48),
      SepResBlock(48),
    )
    self.up4 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
      SepConvGNAct(48, 48),
      SepResBlock(48),
    )
    self.up5 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
      SepConvGNAct(48, 48),
      SepResBlock(48),
    )
    self.head = nn.Sequential(
      SepConvGNAct(48, 32),
      nn.Conv2d(32, 6, 1),
    )

  def forward(self, latent: torch.Tensor) -> torch.Tensor:
    coords = make_coord_grid(latent.shape[0], latent.shape[-2], latent.shape[-1], latent.device)
    x = torch.cat([latent, coords], dim=1)
    x = self.block1(self.stem(x))
    x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)
    x = self.up4(x)
    x = self.up5(x)
    x = torch.sigmoid(self.head(x)) * 255.0
    return einops.rearrange(x, "b (t c) h w -> b t c h w", t=PAIR_LEN, c=3)


def decode_payload(path: Path) -> dict:
  return torch.load(io.BytesIO(brotli.decompress(path.read_bytes())), map_location="cpu")


def restore_model(payload: dict, device: torch.device) -> PairLatentDecoder:
  model = PairLatentDecoder().to(device)
  state = {}
  for name, rec in payload["model"].items():
    q = rec["q"].numpy() if hasattr(rec["q"], "numpy") else rec["q"]
    arr = q.astype(np.float32) * float(rec["scale"])
    state[name] = torch.from_numpy(arr.reshape(tuple(rec["shape"])))
  model.load_state_dict(state, strict=True)
  model.eval()
  return model


def restore_latents(payload: dict) -> torch.Tensor:
  q = payload["latents"].numpy() if hasattr(payload["latents"], "numpy") else payload["latents"]
  arr = q.astype(np.float32) * float(payload["scale"])
  return torch.from_numpy(arr.reshape(tuple(payload["shape"])))


def main() -> None:
  if len(sys.argv) < 4:
    raise SystemExit("Usage: inflate.py <data_dir> <output_dir> <file_list_txt>")

  data_dir = Path(sys.argv[1])
  output_dir = Path(sys.argv[2])
  file_list_path = Path(sys.argv[3])
  output_dir.mkdir(parents=True, exist_ok=True)

  files = [line.strip() for line in file_list_path.read_text().splitlines() if line.strip()]
  if len(files) != 1:
    raise ValueError("latent_pair_codec currently assumes a single video file.")

  device = choose_device()
  model_payload = decode_payload(data_dir / "model.pt.br")
  latent_payload = decode_payload(data_dir / "latents.pt.br")

  model = restore_model(model_payload, device)
  latents = restore_latents(latent_payload).to(device)

  raw_out_path = output_dir / f"{Path(files[0]).stem}.raw"
  with torch.inference_mode(), open(raw_out_path, "wb") as f_out:
    for start in tqdm(range(0, latents.shape[0], 16), desc="inflate latent_pair_codec"):
      z = latents[start : start + 16]
      pred_small = model(z)
      pred_full = F.interpolate(
        einops.rearrange(pred_small, "b t c h w -> (b t) c h w"),
        size=(camera_size[1], camera_size[0]),
        mode="bilinear",
        align_corners=False,
      )
      pred_full = pred_full.clamp(0, 255).round().to(torch.uint8)
      pred_full = einops.rearrange(pred_full, "(b t) c h w -> (b t) h w c", t=PAIR_LEN)
      f_out.write(pred_full.cpu().numpy().tobytes())


if __name__ == "__main__":
  main()
