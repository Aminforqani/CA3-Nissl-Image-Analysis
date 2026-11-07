#!/usr/bin/env python3
"""CA3 Nissl V3 — clean, single-method cell-centre detection.

Output per image: one annotated PNG. Output per run: one concise summary.csv.
Classes are morphology-based estimates; this is not validated stereology.

Usage:
    py ca3_nissl_v3_clean.py img_D.png --out results
    py ca3_nissl_v3_clean.py img_A.png img_C.png --out results
Optional: --roi ca3_mask.png, --export-points
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import color, feature, filters, io, measure, morphology, segmentation
from skimage.transform import resize

TARGET_WIDTH = 1429
REFERENCE_WIDTH = 975
RECENTER_RADIUS = 65.0
NUCLEUS_THRESHOLD = 0.045
PEAK_SPACING = 14
DISTANCE_PENALTY = 0.30

PALETTE = {
    "neuron": (255, 207, 41),
    "glia": (255, 64, 164),
    "uncertain": (8, 217, 233),
}


def read_image(path: Path):
    image = io.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]
    if image.dtype != np.uint8:
        limit = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
        image = np.clip(image.astype(float) / limit * 255, 0, 255).astype(np.uint8)

    original_size = image.shape[:2]
    if image.shape[1] != TARGET_WIDTH:
        height = round(image.shape[0] * TARGET_WIDTH / image.shape[1])
        image = resize(image, (height, TARGET_WIDTH), preserve_range=True,
                       anti_aliasing=True).astype(np.uint8)
    return image, original_size


def read_roi(path: Path | None, shape, original_shape):
    if path is None:
        return None
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=-1)
    if mask.shape == original_shape:
        mask = resize(mask.astype(float), shape, order=0,
                      preserve_range=True, anti_aliasing=False)
    elif mask.shape != shape:
        raise ValueError("ROI dimensions must match the original or resized image")
    return mask > 127


def segment_and_locate(image: np.ndarray, roi=None):
    """Original Nissl optical-density / Otsu / marker watershed stage."""
    scale = image.shape[1] / REFERENCE_WIDTH
    od = -np.log(np.clip(image.astype(np.float32) / 255, 1e-3, 1.0))
    stain = od.mean(axis=2)
    smooth = filters.gaussian(stain, 2 * scale)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, max_size=max(1, int(150 * scale**2)))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.opening(mask, morphology.disk(max(1, round(2 * scale))))
    if roi is not None:
        mask &= roi

    distance = filters.gaussian(ndi.distance_transform_edt(mask), 2 * scale)
    distance_peaks = feature.peak_local_max(
        distance, min_distance=max(3, int(14 * scale)),
        threshold_abs=8 * scale, labels=measure.label(mask)
    )
    surround = filters.gaussian(stain, 30 * scale / 1.466)
    pale = np.clip(surround - smooth, 0, None) * mask
    nuclei = feature.blob_log(
        pale / max(float(pale.max()), 1e-8), min_sigma=15 * scale / 1.466,
        max_sigma=24 * scale / 1.466, num_sigma=4,
        threshold=0.12, overlap=0.1,
    )[:, :2].astype(int)

    centres = []
    for group in (nuclei, distance_peaks):
        for y, x in group:
            if mask[y, x] and not any((y-a)**2 + (x-b)**2 < (32 * scale)**2
                                      for a, b in centres):
                centres.append((int(y), int(x)))
    markers = np.zeros(mask.shape, dtype=np.int32)
    if centres:
        ys, xs = np.asarray(centres).T
        markers[ys, xs] = np.arange(1, len(centres) + 1)
    labels = segmentation.watershed(
        smooth - distance / max(float(distance.max()), 1e-8), markers, mask=mask
    )
    hue = color.rgb2hsv(image)[..., 0]
    profiles = []
    for region in measure.regionprops(labels, intensity_image=np.dstack([stain, hue])):
        if region.area < 150 * scale**2:
            continue
        circularity = 4 * np.pi * region.area / max(region.perimeter, 1)**2
        _, mean_hue = region.intensity_mean
        y, x = region.centroid
        cell_type = ("glia" if region.area <= 1500 * scale**2
                     and circularity > 0.78 and mean_hue < 0.705 else "neuron")
        profiles.append({
            "region": int(region.label), "x": float(x), "y": float(y),
            "area": float(region.area), "circularity": float(circularity),
            "class": cell_type,
        })
    return profiles, labels, stain


def recenter_and_classify(profiles, labels, stain):
    """V3: move centres towards plausible pale nuclei; never add/remove profiles."""
    if not profiles:
        return []
    response = filters.gaussian(stain, 28) - filters.gaussian(stain, 14)
    peaks = feature.peak_local_max(
        response, min_distance=PEAK_SPACING, threshold_abs=0.004,
        exclude_border=False, labels=(labels > 0).astype(np.uint8)
    )
    by_region = {}
    for y, x in peaks:
        region = int(labels[y, x])
        if region:
            by_region.setdefault(region, []).append((float(x), float(y), float(response[y, x])))

    results = []
    for profile in profiles:
        cell = profile.copy()
        candidates = by_region.get(cell["region"], [])
        if candidates:
            options = np.asarray(candidates, dtype=float)
            distances = np.hypot(options[:, 0] - cell["x"], options[:, 1] - cell["y"])
            allowed = (distances < RECENTER_RADIUS) & (options[:, 2] > NUCLEUS_THRESHOLD)
            if np.any(allowed):
                quality = (options[:, 2] / (np.quantile(options[:, 2], 0.95) + 1e-6)
                           - DISTANCE_PENALTY * distances / RECENTER_RADIUS)
                index = int(np.argmax(np.where(allowed, quality, -1e8)))
                cell["x"], cell["y"] = options[index, :2].tolist()

        # This uncertain class is an abstention, not a separate cell identity.
        if cell["circularity"] < 0.45 or (cell["area"] < 2200 and cell["class"] != "glia"):
            cell["class"] = "uncertain"
        results.append(cell)
    return results


def render(image, cells, path):
    """One clean, full-resolution annotated image; no plots or duplicate panels."""
    count = {name: sum(cell["class"] == name for cell in cells) for name in PALETTE}
    header = 64
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + header), "white")
    canvas.paste(Image.fromarray(image), (0, header))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    draw.text((16, 18), "CA3 Nissl  |  V3", fill=(35, 40, 50), font=font)
    positions = {"neuron": 410, "glia": 750, "uncertain": 1040}
    for name, x in positions.items():
        rgb = PALETTE[name]
        draw.ellipse((x, 26, x + 14, 40), fill=rgb, outline=(25, 25, 25), width=1)
        draw.text((x + 24, 18), f"{name}: {count[name]}", fill=(35, 40, 50), font=font)

    for cell in cells:
        x, y = round(cell["x"]), round(cell["y"] + header)
        rgb = PALETTE[cell["class"]]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=rgb, outline=(20, 20, 20), width=1)
    canvas.save(path)
    return count


def analyze(images, output, roi=None, export_points=False):
    output.mkdir(parents=True, exist_ok=True)
    summary, point_rows = [], []
    for file in images:
        image, shape = read_image(file)
        region = read_roi(roi, image.shape[:2], shape)
        profiles, labels, stain = segment_and_locate(image, region)
        cells = recenter_and_classify(profiles, labels, stain)
        png = output / f"{file.stem}_V3.png"
        counts = render(image, cells, png)
        summary.append({"image": file.name, "neurons": counts["neuron"],
                        "glia": counts["glia"], "uncertain": counts["uncertain"],
                        "total": len(cells)})
        if export_points:
            for cell in cells:
                point_rows.append({"image": file.name, "x": round(cell["x"]),
                                   "y": round(cell["y"]), "class": cell["class"]})
        print(f"{file.name}: {len(cells)} candidates | output: {png}")

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["image", "neurons", "glia", "uncertain", "total"])
        writer.writeheader()
        writer.writerows(summary)
    if export_points:
        with (output / "cell_points.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=["image", "x", "y", "class"])
            writer.writeheader()
            writer.writerows(point_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="CA3 Nissl V3 — clean single-method output")
    parser.add_argument("images", type=Path, nargs="+", help="Original, unannotated Nissl images")
    parser.add_argument("--out", type=Path, default=Path("results_V3"))
    parser.add_argument("--roi", type=Path, default=None, help="Optional white-on-black ROI mask")
    parser.add_argument("--export-points", action="store_true", help="Additionally save x,y,class CSV")
    args = parser.parse_args()
    for image in args.images:
        if not image.is_file():
            parser.error(f"Image not found: {image}")
    analyze(args.images, args.out, args.roi, args.export_points)


if __name__ == "__main__":
    main()