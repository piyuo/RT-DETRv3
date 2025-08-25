# tools/build_calib_images.py
from pathlib import Path
from PIL import Image, ImageOps

input_dir = Path("calib/input")
output_dir = Path("calib/output")
output_dir.mkdir(parents=True, exist_ok=True)

target_size = (640, 640)

for i, img_path in enumerate(input_dir.glob("*.png")):
    img = Image.open(img_path).convert("RGB")

    # Resize UP or DOWN with aspect ratio preserved, max within 640x640
    img = ImageOps.contain(img, target_size, Image.Resampling.LANCZOS)

    # Pad to exactly 640x640 with gray background (114,114,114)
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    img = ImageOps.expand(img, padding, fill=(114, 114, 114))

    img.save(output_dir / f"{i:04d}.png")
