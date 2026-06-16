#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supp Fig S3 (S2A/S2B): stitch existing SHAP bar-summary PNGs into a clean 1x2 panel,
and FIX the x-axis label clipping problem by:
1) adding EXTRA bottom margin (so labels have room),
2) drawing an opaque WHITE rectangle behind each x-label zone (so nothing looks cut),
3) writing a SHORTENED x-label ourselves (so it never clips).

No SHAP recomputation. Just image-level clean panelization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


# --------------------------
# helpers
# --------------------------
def try_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in [
        "Arial Bold.ttf",
        "Arial Bold",
        "Arial.ttf",
        "Helvetica.ttc",
        "Helvetica",
    ]:
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            pass
    return ImageFont.load_default()


def trim_whitespace(im: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """Auto-trim near-white borders. Keeps content; avoids shaving axes by padding back."""
    im = im.convert("RGB")
    bg_im = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_im)
    diff = ImageChops.add(diff, diff, 2.0, -18)
    bbox = diff.getbbox()
    if not bbox:
        return im
    cropped = im.crop(bbox)

    pad = 8
    out = Image.new("RGB", (cropped.size[0] + 2 * pad, cropped.size[1] + 2 * pad), bg)
    out.paste(cropped, (pad, pad))
    return out


def load_img(p: Path) -> Image.Image:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return trim_whitespace(Image.open(p))


def resize_to_same_height(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    ha, hb = a.size[1], b.size[1]
    target_h = min(ha, hb)

    def r(im: Image.Image) -> Image.Image:
        w, h = im.size
        if h == target_h:
            return im
        new_w = int(round(w * (target_h / h)))
        return im.resize((new_w, target_h), Image.LANCZOS)

    return r(a), r(b)


def draw_panel_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, px: int):
    font = try_font(px)
    stroke = max(1, px // 14)
    draw.text(
        (x, y),
        text,
        fill=(0, 0, 0),
        font=font,
        stroke_width=stroke,
        stroke_fill=(255, 255, 255),
    )


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    # robust bbox sizing (handles different PIL versions/fonts)
    bb = draw.textbbox((0, 0), text, font=font)
    return (bb[2] - bb[0], bb[3] - bb[1])


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poc_root", default="data/processed")
    ap.add_argument("--out_dir", default="figures/supplementary")
    ap.add_argument("--left_png", default="", help="S2A source PNG")
    ap.add_argument("--right_png", default="", help="S2B source PNG")
    ap.add_argument("--dpi", type=int, default=600)

    # layout
    ap.add_argument("--outer_pad", type=int, default=22)
    ap.add_argument("--gap", type=int, default=40)
    ap.add_argument("--extra_bottom", type=int, default=130, help="extra bottom space for rebuilt labels + panel tags")

    # label styling
    ap.add_argument("--panel_label_px", type=int, default=58)
    ap.add_argument("--xlabel_px", type=int, default=28)
    ap.add_argument(
        "--xlabel_text",
        default="mean(|SHAP|) (avg. impact on output)",
        help="short label (won't clip)",
    )
    ap.add_argument("--xlabel_box_h", type=int, default=86, help="white box height behind x-label")
    ap.add_argument("--xlabel_box_margin", type=int, default=6, help="white box side margin inside each panel")

    # stronger cover of old clipped label region (inside panel image)
    ap.add_argument("--xlabel_cover_overlap", type=int, default=52, help="how far the white box intrudes into panel image")
    args = ap.parse_args()

    poc_root = Path(args.poc_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_left = poc_root / "실험 결과값들" / "실험 그래프들" / "seed42" / "explainer" / "shap s.png"
    default_right = poc_root / "실험 결과값들" / "실험 그래프들" / "seed42" / "explainer" / "shap sc.png"
    left_path = Path(args.left_png) if args.left_png else default_left
    right_path = Path(args.right_png) if args.right_png else default_right

    left = load_img(left_path)
    right = load_img(right_path)
    left, right = resize_to_same_height(left, right)

    wL, h = left.size
    wR, _ = right.size

    bg = (255, 255, 255)
    pad = args.outer_pad
    gap = args.gap
    extra_bottom = args.extra_bottom

    # Final canvas
    canvas_w = pad + wL + gap + wR + pad
    canvas_h = pad + h + extra_bottom + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    # Paste panels (top aligned)
    xL = pad
    xR = pad + wL + gap
    y0 = pad
    canvas.paste(left, (xL, y0))
    canvas.paste(right, (xR, y0))

    # --- FIX X-AXIS LABEL CLIPPING ---
    # Cover the old x-label region with an opaque white box that intrudes into the panel image
    xlabel_font = try_font(args.xlabel_px)

    # box sits overlapping the bottom of each panel image to fully hide clipped remnants
    box_h = args.xlabel_box_h
    overlap = args.xlabel_cover_overlap  # deeper overlap to fully cover original x-label remnants
    box_y = y0 + h - overlap

    # Box bounds per panel (almost full width)
    L_box_x0 = xL + args.xlabel_box_margin
    L_box_x1 = xL + wL - args.xlabel_box_margin
    R_box_x0 = xR + args.xlabel_box_margin
    R_box_x1 = xR + wR - args.xlabel_box_margin

    # Draw white cover rectangles
    draw.rectangle([L_box_x0, box_y, L_box_x1, box_y + box_h], fill=bg)
    draw.rectangle([R_box_x0, box_y, R_box_x1, box_y + box_h], fill=bg)

    # Centered x-label text inside each box (true centered)
    def centered_text(box_x0: int, box_x1: int, text: str):
        tw, th = text_size(draw, text, xlabel_font)
        cx = (box_x0 + box_x1) / 2.0
        tx = int(round(cx - tw / 2.0))
        ty = int(round(box_y + (box_h - th) / 2.0))
        draw.text((tx, ty), text, fill=(0, 0, 0), font=xlabel_font)

    centered_text(L_box_x0, L_box_x1, args.xlabel_text)
    centered_text(R_box_x0, R_box_x1, args.xlabel_text)

    # --- Panel labels (S2A/S2B): place in the SAFE bottom margin area so they never clip ---
    # Put them slightly above the bottom pad, aligned to each panel's left edge.
    panel_font_px = args.panel_label_px
    panel_font = try_font(panel_font_px)

    label_y = y0 + h + (extra_bottom // 2)  # middle of the extra bottom area
    # nudge upward by half text height
    _, lbl_th = text_size(draw, "S2A", panel_font)
    label_y = int(round(label_y - lbl_th / 2))

    draw_panel_label(draw, xL + 6, label_y, "S2A", panel_font_px)
    draw_panel_label(draw, xR + 6, label_y, "S2B", panel_font_px)

    # Save outputs
    out_png = out_dir / "SuppFig_S3.png"
    out_jpg = out_dir / "SuppFig_S3.jpg"
    out_pdf = out_dir / "SuppFig_S3.pdf"

    canvas.save(out_png, dpi=(args.dpi, args.dpi))
    canvas.save(out_jpg, quality=95, subsampling=0, dpi=(args.dpi, args.dpi))
    canvas.save(out_pdf, "PDF", resolution=args.dpi)

    print("[OK] S2A:", left_path)
    print("[OK] S2B:", right_path)
    print("[OK] Saved:", out_png)
    print("[OK] Saved:", out_jpg)
    print("[OK] Saved:", out_pdf)


if __name__ == "__main__":
    main()
