"""C15 image pair set against the REAL Qwen-Image VAE, inside the ComfyUI runtime image.

Lane M2 (2026-10-10). Sibling of ``c15_vae_pairs_container.py`` for the IMAGE band: one row
per Qwen-Image DiT token, i.e. a 2x2 group of latent pixels (16 channels -> 64 features,
packed ``(c, ph, pw)`` exactly as ComfyUI ``comfy/ldm/qwen_image/model.py::process_img``
packs ``img_in``'s input) paired with the 16x16x3 pixel patch it covers (768 raw values,
packed ``(c, y, x)``) and with the parent VAE's own round trip of that patch.

Latent space: the DiT's own (``comfy.latent_formats.Wan21().process_in``, the format
ComfyUI's QwenImage model uses), because that is what ``img_in`` reads. The C15 PSNR is
invariant to that per-channel affine change.

Images: real images on the box (``--images-dir``), chosen deterministically (sorted file
names, evenly spaced), centre-cropped square and resized to ``--size``. Held-out split is
BY IMAGE. Nothing is generated and no DiT is constructed: only the VAE runs.

    python c15_image_pairs_container.py --vae /models/vae/qwen_image_vae.safetensors \
        --images-dir /images --count 48 --size 256 --out /work/pairs/qimg_vae_tok2 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for _comfy_root in ("/opt/ComfyUI",):
    if Path(_comfy_root).is_dir() and _comfy_root not in sys.path:
        sys.path.append(_comfy_root)

import torch  # noqa: E402

from mediafold.c15_pairs import PairSet, _clip_split, write_pair_set  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def choose_images(directory: Path, count: int, min_side: int) -> list[Path]:
    from PIL import Image

    candidates = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    usable: list[Path] = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                if min(image.size) >= int(min_side):
                    usable.append(path)
        except Exception:
            continue
    if len(usable) < count:
        raise SystemExit(f"only {len(usable)} usable images >= {min_side}px under {directory}; need {count}")
    step = len(usable) / float(count)
    return [usable[int(i * step)] for i in range(count)]


def load_square(path: Path, size: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(size, size, 3)
    return data.to(torch.float32) / 255.0  # [H, W, 3] in [0, 1]


def pack_pixels(image_hwc: torch.Tensor, patch: int) -> torch.Tensor:
    """[H, W, 3] -> [tokens, 3*patch*patch] packed (c, y, x), tokens row-major over (h, w)."""

    h, w, c = image_hwc.shape
    chw = image_hwc.permute(2, 0, 1)
    grid = chw.reshape(c, h // patch, patch, w // patch, patch).permute(1, 3, 0, 2, 4)
    return grid.reshape((h // patch) * (w // patch), c * patch * patch)


def pack_latent(latent_chw: torch.Tensor) -> torch.Tensor:
    """[16, h, w] -> [tokens, 64] packed (c, ph, pw) -- ComfyUI process_img order."""

    c, h, w = latent_chw.shape
    grid = latent_chw.reshape(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4)
    return grid.reshape((h // 2) * (w // 2), c * 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C15 image pairs against the Qwen-Image VAE")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.size % 16:
        raise SystemExit("--size must be a multiple of 16 (8x VAE stride x 2x2 DiT patch)")
    if str(args.device).startswith("cpu"):
        # ComfyUI picks its device at import from cli_args; without --cpu it asks CUDA
        import comfy.cli_args

        comfy.cli_args.args.cpu = True
    import comfy.latent_formats
    import comfy.sd
    import comfy.utils

    torch.manual_seed(int(args.seed))
    vae_path = Path(args.vae)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(vae_path)))
    latent_format = comfy.latent_formats.Wan21()
    images = choose_images(Path(args.images_dir), int(args.count), int(args.size))

    raw_rows, latent_rows, roundtrip_rows = [], [], []
    clip_index, frame_index = [], []
    per_image = []
    for index, path in enumerate(images):
        pixels = load_square(path, int(args.size))  # [H, W, 3]
        with torch.no_grad():
            latent = vae.encode(pixels.unsqueeze(0))  # comfy: [B, H, W, C] in [0, 1]
            decoded = vae.decode(latent)
        latent = latent.detach().float().cpu()
        decoded = decoded.detach().float().cpu().reshape(-1, int(args.size), int(args.size), 3)[0].clamp(0.0, 1.0)
        if latent.ndim == 4:  # [B, C, h, w] -> [B, C, 1, h, w]
            latent = latent.unsqueeze(2)
        dit_latent = latent_format.process_in(latent)[0, :, 0]  # [16, h, w]
        if dit_latent.shape[0] != 16 or dit_latent.shape[1] * 8 != int(args.size):
            raise SystemExit(f"unexpected latent shape {tuple(latent.shape)} for {path}")
        raw = pack_pixels(pixels, 16)
        rt = pack_pixels(decoded, 16)
        lat = pack_latent(dit_latent)
        tokens = raw.shape[0]
        raw_rows.append(raw)
        roundtrip_rows.append(rt)
        latent_rows.append(lat)
        clip_index.append(torch.full((tokens,), index, dtype=torch.long))
        frame_index.append(torch.arange(tokens, dtype=torch.long))
        mse = float((decoded - pixels).pow(2).mean())
        per_image.append(
            {
                "index": index,
                "file": str(path),
                "sha256": sha256_file(path),
                "tokens": int(tokens),
                "vae_roundtrip_psnr_db": float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12)))),
            }
        )
        print(f"[{index + 1}/{len(images)}] {path.name}: tokens {tokens}, round trip {per_image[-1]['vae_roundtrip_psnr_db']:.2f} dB", flush=True)

    split_per_image = _clip_split(len(images), float(args.holdout_fraction), int(args.seed))
    clip = torch.cat(clip_index)
    pairs = PairSet(
        raw=torch.cat(raw_rows),
        latent=torch.cat(latent_rows),
        roundtrip=torch.cat(roundtrip_rows),
        clip_index=clip,
        frame_index=torch.cat(frame_index),
        channel_index=torch.zeros_like(clip),
        split=split_per_image[clip],
        geometry={
            "name": vae_path.name,
            "modality": "image",
            "latent_channels": 16,
            "spatial_stride": 8,
            "latent_group": 2,
            "pixel_range": [0.0, 1.0],
            "image_size": int(args.size),
            "d_raw": 768,
            "d_latent": 64,
            "d_centre": 768,
            "centre_offset": 0,
            "context": 0,
            "rows": int(clip.numel()),
            "clips": len(images),
            "raw_packing": "(c, y, x) over one 16x16 RGB patch",
            "latent_packing": "(c, ph, pw) = ComfyUI qwen_image process_img order (img_in input)",
            "latent_space": "DiT space: comfy.latent_formats.Wan21().process_in(vae.encode(x))",
            "pairing": "one row per Qwen-Image DiT token",
        },
    )
    receipt = write_pair_set(
        pairs,
        args.out,
        extra={
            "images": per_image,
            "vae": {"path": str(vae_path), "sha256": sha256_file(vae_path)},
            "selection": {"images_dir": str(args.images_dir), "rule": "sorted names, >= size px short side, evenly spaced", "count": len(images)},
            "split_images": {"holdout": [int(i) for i in torch.nonzero(split_per_image).flatten()], "seed": int(args.seed)},
            "argv": vars(args),
            "lane": "M2 C15 image band (2026-10-10)",
        },
    )
    print(json.dumps({"rows": receipt["pairs"]["parameters"], "split": receipt["split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
