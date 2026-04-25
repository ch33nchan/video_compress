#!/usr/bin/env python3
import io
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb
from modules import DistortionNet, SegNet, PoseNet, posenet_sd_path, segnet_sd_path

LOWRES_W = 512
LOWRES_H = 384
PAIR_LEN = 2
LATENT_C = 2
LATENT_H = 12
LATENT_W = 16
TRAIN_SEED = 1234


def choose_device(device_arg: str | None) -> torch.device:
  if device_arg:
    return torch.device(device_arg)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def round_ste(x: torch.Tensor) -> torch.Tensor:
  return x + (x.round() - x).detach()


def quantize_int8_ste(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  q = torch.clamp(round_ste(x / scale), -127.0, 127.0)
  return q * scale


def decode_video_frames(video_path: Path) -> torch.Tensor:
  container = av.open(str(video_path))
  stream = container.streams.video[0]
  frames = []
  for frame in container.decode(stream):
    frames.append(yuv420_to_rgb(frame))
  container.close()
  return torch.stack(frames, dim=0)


def make_frame_pairs(frames: torch.Tensor) -> torch.Tensor:
  pair_count = frames.shape[0] // PAIR_LEN
  frames = frames[: pair_count * PAIR_LEN]
  return einops.rearrange(frames, "(n t) h w c -> n t h w c", t=PAIR_LEN)


def make_coord_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
  ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
  xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
  yy, xx = torch.meshgrid(ys, xs, indexing="ij")
  grid = torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0)
  return grid.unsqueeze(0).expand(batch, -1, -1, -1)


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


@dataclass
class QuantizedTensor:
  data: np.ndarray
  scale: float
  shape: tuple[int, ...]


def quantize_numpy_int8(x: np.ndarray) -> QuantizedTensor:
  max_abs = float(np.max(np.abs(x)))
  scale = max(max_abs / 127.0, 1e-8)
  q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
  return QuantizedTensor(data=q, scale=scale, shape=x.shape)


def encode_payload(payload: dict) -> bytes:
  buffer = io.BytesIO()
  torch.save(payload, buffer, _use_new_zipfile_serialization=False)
  return brotli.compress(buffer.getvalue(), quality=11, lgwin=24)


def export_model_state(model: nn.Module) -> dict:
  state = {}
  for name, tensor in model.state_dict().items():
    arr = tensor.detach().cpu().float().numpy()
    q = quantize_numpy_int8(arr)
    state[name] = {"q": q.data, "scale": q.scale, "shape": q.shape}
  return state


def preprocess_targets(pair_rgb: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  gt_full = pair_rgb.to(device).float()
  gt_small = einops.rearrange(gt_full, "b t h w c -> (b t) c h w")
  gt_small = F.interpolate(gt_small, size=(LOWRES_H, LOWRES_W), mode="bilinear", align_corners=False)
  gt_small = einops.rearrange(gt_small, "(b t) c h w -> b t c h w", t=PAIR_LEN)
  return gt_full, gt_small


def run_training(
  pair_rgb: torch.Tensor,
  device: torch.device,
  epochs: int,
  batch_size: int,
  latent_l1: float,
  rgb_weight: float,
  seg_weight: float,
  pose_weight: float,
) -> tuple[PairLatentDecoder, torch.Tensor]:
  torch.manual_seed(TRAIN_SEED)
  np.random.seed(TRAIN_SEED)

  dist = DistortionNet().eval().to(device)
  dist.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  for p in dist.parameters():
    p.requires_grad = False
  segnet: SegNet = dist.segnet
  posenet: PoseNet = dist.posenet

  decoder = PairLatentDecoder().to(device)
  pair_count = pair_rgb.shape[0]
  latent = nn.Parameter(torch.randn(pair_count, LATENT_C, LATENT_H, LATENT_W, device=device) * 0.05)
  latent_scale = nn.Parameter(torch.tensor(0.10, device=device))

  params = list(decoder.parameters()) + [latent, latent_scale]
  optimizer = torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.95))
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-4)

  indices = torch.arange(pair_count)
  best_metric = float("inf")
  best_state = None
  best_latent = None

  for epoch in range(epochs):
    perm = indices[torch.randperm(pair_count)]
    decoder.train()
    running = {"loss": 0.0, "rgb": 0.0, "seg": 0.0, "pose": 0.0}
    batches = 0
    qat_on = epoch >= max(10, epochs // 4)
    current_scale = latent_scale.abs().clamp_min(1e-4)

    pbar = tqdm(range(0, pair_count, batch_size), desc=f"latent_pair_codec epoch {epoch + 1}/{epochs}", leave=False)
    for start in pbar:
      batch_idx_cpu = perm[start : start + batch_size]
      batch_idx_dev = batch_idx_cpu.to(device)
      gt_full, gt_small = preprocess_targets(pair_rgb.index_select(0, batch_idx_cpu), device)

      z = latent.index_select(0, batch_idx_dev)
      z_q = quantize_int8_ste(z, current_scale) if qat_on else z
      pred_small = decoder(z_q)
      pred_full = F.interpolate(
        einops.rearrange(pred_small, "b t c h w -> (b t) c h w"),
        size=(camera_size[1], camera_size[0]),
        mode="bilinear",
        align_corners=False,
      )
      pred_full = einops.rearrange(pred_full, "(b t) c h w -> b t h w c", t=PAIR_LEN)
      pred_small_bchw = pred_small
      pred_small_bhwc = einops.rearrange(pred_small_bchw, "b t c h w -> b t h w c")

      rgb_loss = F.l1_loss(pred_small_bchw, gt_small)

      with torch.no_grad():
        gt_small_bhwc = einops.rearrange(gt_small, "b t c h w -> b t h w c")
        seg_gt_in = segnet.preprocess_input(gt_small)
        seg_gt_logits = segnet(seg_gt_in).float()
        pose_gt = posenet(posenet.preprocess_input(gt_small)).get("pose").float()[..., :6]

      seg_pred_in = segnet.preprocess_input(pred_small_bchw)
      seg_pred_logits = segnet(seg_pred_in).float()
      seg_loss = F.kl_div(F.log_softmax(seg_pred_logits, dim=1), F.softmax(seg_gt_logits, dim=1), reduction="batchmean")

      pose_pred = posenet(posenet.preprocess_input(pred_small_bchw)).get("pose").float()[..., :6]
      pose_loss = F.mse_loss(pose_pred, pose_gt)

      latent_penalty = z_q.abs().mean()
      loss = rgb_weight * rgb_loss + seg_weight * seg_loss + pose_weight * pose_loss + latent_l1 * latent_penalty

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
      optimizer.step()

      running["loss"] += float(loss.item())
      running["rgb"] += float(rgb_loss.item())
      running["seg"] += float(seg_loss.item())
      running["pose"] += float(pose_loss.item())
      batches += 1
      pbar.set_postfix(loss=f"{running['loss'] / batches:.4f}", seg=f"{running['seg'] / batches:.4f}", pose=f"{running['pose'] / batches:.5f}")

    scheduler.step()

    if qat_on and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
      decoder.eval()
      with torch.inference_mode():
        z_eval = quantize_int8_ste(latent, latent_scale.abs().clamp_min(1e-4))
        pred_small = decoder(z_eval)
        pred_full = F.interpolate(
          einops.rearrange(pred_small, "b t c h w -> (b t) c h w"),
          size=(camera_size[1], camera_size[0]),
          mode="bilinear",
          align_corners=False,
        )
        pred_full = einops.rearrange(pred_full, "(b t) c h w -> b t h w c", t=PAIR_LEN).clamp(0, 255).round().to(torch.uint8)
        gt_uint8 = pair_rgb.to(device)
        pose_dist, seg_dist = dist.compute_distortion(gt_uint8, pred_full)
        model_payload = encode_payload({"model": export_model_state(decoder)})
        latent_payload = encode_payload(
          {
            "latents": quantize_numpy_int8(z_eval.detach().cpu().numpy()).data,
            "scale": float(latent_scale.abs().clamp_min(1e-4).detach().cpu()),
            "shape": tuple(z_eval.shape),
          }
        )
        total_bytes = len(model_payload) + len(latent_payload)
        rate = total_bytes / sum(file.stat().st_size for file in (ROOT / "videos").rglob("*") if file.is_file())
        metric = 100.0 * float(seg_dist.mean()) + math.sqrt(10.0 * float(pose_dist.mean())) + 25.0 * rate
        if metric < best_metric:
          best_metric = metric
          best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
          best_latent = z_eval.detach().cpu().clone()
        print(
          f"[eval] epoch={epoch + 1} metric={metric:.4f} seg={float(seg_dist.mean()):.6f} "
          f"pose={float(pose_dist.mean()):.6f} bytes={total_bytes}"
        )

  if best_state is None:
    best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    best_latent = quantize_int8_ste(latent, latent_scale.abs().clamp_min(1e-4)).detach().cpu().clone()

  decoder.load_state_dict(best_state)
  return decoder.cpu(), best_latent


def write_archive(submission_dir: Path, decoder: PairLatentDecoder, latents: torch.Tensor) -> None:
  archive_dir = submission_dir / "archive"
  if archive_dir.exists():
    shutil.rmtree(archive_dir)
  archive_dir.mkdir(parents=True, exist_ok=True)

  model_payload = encode_payload({"model": export_model_state(decoder)})
  latent_quant = quantize_numpy_int8(latents.float().numpy())
  latent_payload = encode_payload({"latents": latent_quant.data, "scale": latent_quant.scale, "shape": latent_quant.shape})

  (archive_dir / "model.pt.br").write_bytes(model_payload)
  (archive_dir / "latents.pt.br").write_bytes(latent_payload)

  archive_zip = submission_dir / "archive.zip"
  if archive_zip.exists():
    archive_zip.unlink()
  with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.write(archive_dir / "model.pt.br", arcname="model.pt.br")
    zf.write(archive_dir / "latents.pt.br", arcname="latents.pt.br")

  total_bytes = archive_zip.stat().st_size
  print(f"wrote {archive_zip} ({total_bytes} bytes)")


def parse_args():
  import argparse

  parser = argparse.ArgumentParser(description="Train a tiny pair-latent codec submission.")
  parser.add_argument("--video-dir", type=Path, default=ROOT / "videos")
  parser.add_argument("--video-names", type=Path, default=ROOT / "public_test_video_names.txt")
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--latent-l1", type=float, default=2e-4)
  parser.add_argument("--rgb-weight", type=float, default=1.0)
  parser.add_argument("--seg-weight", type=float, default=0.25)
  parser.add_argument("--pose-weight", type=float, default=4.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = choose_device(args.device)
  files = [line.strip() for line in args.video_names.read_text().splitlines() if line.strip()]
  if len(files) != 1:
    raise ValueError("latent_pair_codec currently assumes a single video file.")
  video_path = args.video_dir / files[0]

  print(f"device={device}")
  print(f"loading {video_path}")
  frames = decode_video_frames(video_path)
  pair_rgb = make_frame_pairs(frames)
  print(f"loaded {frames.shape[0]} frames -> {pair_rgb.shape[0]} pairs")

  decoder, latents = run_training(
    pair_rgb=pair_rgb,
    device=device,
    epochs=args.epochs,
    batch_size=args.batch_size,
    latent_l1=args.latent_l1,
    rgb_weight=args.rgb_weight,
    seg_weight=args.seg_weight,
    pose_weight=args.pose_weight,
  )
  write_archive(Path(__file__).resolve().parent, decoder, latents)


if __name__ == "__main__":
  main()
