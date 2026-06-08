#!/usr/bin/env python3
from pathlib import Path
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def ensure(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def gen_images():
    out = ROOT / "data" / "images"
    ensure(out)
    random.seed(7)
    for i in range(1, 9):
        w, h = 640, 420
        img = Image.new("RGB", (w, h), (205, 205, 205))
        draw = ImageDraw.Draw(img)
        # part body
        draw.rounded_rectangle([60, 50, 580, 360], radius=18, fill=(178, 178, 178), outline=(80, 80, 80), width=2)
        # texture lines
        for x in range(90, 560, 35):
            draw.line([x, 70, x + random.randint(-8, 8), 345], fill=(160, 160, 160), width=1)
        for y in range(80, 340, 40):
            draw.line([80, y, 560, y + random.randint(-3, 3)], fill=(190, 190, 190), width=1)
        # dark defects
        for k in range(random.randint(2, 5)):
            x = random.randint(100, 500)
            y = random.randint(90, 310)
            rw = random.randint(20, 90)
            rh = random.randint(5, 35)
            angle = random.randint(-20, 20)
            # draw ellipse/rectangle-ish defect
            draw.ellipse([x, y, x + rw, y + rh], fill=(35 + random.randint(0, 30),) * 3)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        img.save(out / f"part_{i:03d}.png")


def gen_stereo():
    out = ROOT / "data" / "stereo"
    ensure(out)
    w, h = 320, 240
    base = Image.new("L", (w, h), 120)
    draw = ImageDraw.Draw(base)
    random.seed(11)
    for _ in range(80):
        x = random.randint(10, w - 30)
        y = random.randint(10, h - 30)
        r = random.randint(3, 12)
        c = random.randint(80, 220)
        draw.ellipse([x, y, x + r, y + r], fill=c)
    left = base
    right = Image.new("L", (w, h), 120)
    # shift scene to simulate disparity
    right.paste(base.crop((8, 0, w, h)), (0, 0))
    left.save(out / "left.png")
    right.save(out / "right.png")


def gen_pointcloud():
    out = ROOT / "data" / "pointcloud"
    ensure(out)
    random.seed(13)
    path = out / "plane.xyz"
    with path.open("w", encoding="utf-8") as f:
        for ix in range(-40, 41):
            for iy in range(-30, 31):
                x = ix * 0.5
                y = iy * 0.5
                z = 0.03 * x - 0.02 * y + 10.0 + random.uniform(-0.015, 0.015)
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    gen_images()
    gen_stereo()
    gen_pointcloud()
    print("Generated sample data under data/")


if __name__ == "__main__":
    main()
