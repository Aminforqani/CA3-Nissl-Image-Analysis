"""Automated segmentation of Nissl-stained brain sections.

Steps:
1. Convert RGB to a single "stain" channel (optical density).
2. Smooth and threshold (Otsu) to separate stained cell bodies from neuropil.
3. Clean the mask (remove specks, fill nuclei holes).
4. Split touching cells with a distance-transform watershed.
5. Classify objects: small, round, dark = glia; large = neuronal somata.
6. Report counts, soma size and neuropil fraction; save an overlay figure.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io, color, filters, morphology, measure, segmentation, feature


def stain_channel(rgb):
    """Optical density: dark (stained) pixels -> high values."""
    rgb = np.clip(rgb.astype(float) / 255.0, 1e-3, 1)
    od = -np.log(rgb)            # Beer-Lambert: absorbance per colour channel
    return od.mean(axis=2)


REF_WIDTH = 975  # parameters below were tuned on a 975-px-wide image


def segment(img_path, min_obj_px=150, glia_max_px=1500, sigma=2.0, target_width=1429):
    rgb = io.imread(img_path)[..., :3]
    if rgb.shape[1] != target_width:           # bring every image to the same pixel size
        from skimage.transform import resize
        h = round(rgb.shape[0] * target_width / rgb.shape[1])
        rgb = (resize(rgb, (h, target_width), anti_aliasing=True) * 255).astype(np.uint8)
    k = rgb.shape[1] / REF_WIDTH                 # linear scale factor
    min_obj_px, glia_max_px, sigma = min_obj_px * k**2, glia_max_px * k**2, sigma * k
    od = stain_channel(rgb)
    od_s = filters.gaussian(od, sigma=sigma)

    thr = filters.threshold_otsu(od_s)
    mask = od_s > thr
    mask = morphology.remove_small_objects(mask, max_size=min_obj_px)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.opening(mask, morphology.disk(round(2 * k)))

    # --- seeds: one marker per cell ---------------------------------------
    # (a) pale nuclei inside dark Nissl cytoplasm: works in densely packed layers
    surround = filters.gaussian(od, 30 * k / 1.466)
    contrast = np.clip(surround - od_s, 0, None) * mask
    nuclei = feature.blob_log(contrast / contrast.max(), min_sigma=15 * k / 1.466,
                              max_sigma=24 * k / 1.466, num_sigma=4,
                              threshold=0.12, overlap=0.1)[:, :2].astype(int)
    # (b) peaks of the distance map: works for isolated or dark cells without a visible nucleus
    dist = filters.gaussian(ndi.distance_transform_edt(mask), 2 * k)
    peaks = feature.peak_local_max(dist, min_distance=int(14 * k), threshold_abs=8 * k,
                                   labels=measure.label(mask))
    # keep a distance peak only if no nucleus seed is nearby (avoid two seeds per cell)
    near = 32 * k
    if len(nuclei):
        d = np.sqrt(((peaks[:, None, :] - nuclei[None, :, :]) ** 2).sum(-1)).min(1)
        peaks = peaks[d > near]
    seeds = np.vstack([nuclei, peaks]) if len(nuclei) else peaks
    markers = np.zeros(mask.shape, dtype=int)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    markers[~mask] = 0
    # flood from the seeds over the stain image: neighbouring cells meet between nuclei
    labels = segmentation.watershed(od_s - dist / dist.max(), markers, mask=mask)
    hue = color.rgb2hsv(rgb)[..., 0]

    rows = []
    stack = np.dstack([od, hue])                 # per-object mean OD and hue in one pass
    for r in measure.regionprops(labels, intensity_image=stack):
        if r.area < min_obj_px:
            continue
        circ = 4 * np.pi * r.area / max(r.perimeter, 1) ** 2
        obj_od, obj_hue = r.intensity_mean
        # glia: small, round, and bluer than neuronal Nissl substance
        is_glia = r.area <= glia_max_px and circ > 0.78 and obj_hue < 0.705
        rows.append(dict(label=r.label, area=r.area, circ=round(circ, 2),
                         od=obj_od, hue=obj_hue,
                         cls="glia" if is_glia else "neuron"))

    neuron_ids = [r["label"] for r in rows if r["cls"] == "neuron"]
    glia_ids = [r["label"] for r in rows if r["cls"] == "glia"]
    total_px = mask.size
    neuron_px = np.isin(labels, neuron_ids).sum()
    glia_px = np.isin(labels, glia_ids).sum()

    neuropil_od = od[labels == 0].mean()
    for r in rows:
        r["rel_od"] = r["od"] / neuropil_od      # corrects for batch/illumination differences
    neuron_rel = [r["rel_od"] for r in rows if r["cls"] == "neuron"]
    summary = dict(
        image=Path(img_path).name,
        threshold=round(float(thr), 3),
        n_neurons=len(neuron_ids),
        n_glia=len(glia_ids),
        mean_neuron_area_px=round(float(np.mean([r["area"] for r in rows if r["cls"] == "neuron"])), 1),
        neuron_soma_fraction=round(neuron_px / total_px, 3),
        glia_fraction=round(glia_px / total_px, 3),
        neuropil_fraction=round(1 - (neuron_px + glia_px) / total_px, 3),
        neuropil_od=round(float(neuropil_od), 3),
        mean_neuron_rel_od=round(float(np.mean(neuron_rel)), 2),
    )
    return rgb, od, labels, rows, summary


def outlined(rgb, labels, rows):
    """Return a copy of the image with neuron outlines yellow and glia outlines cyan."""
    out = rgb.copy()
    glia = np.isin(labels, [r["label"] for r in rows if r["cls"] == "glia"])
    neur = np.isin(labels, [r["label"] for r in rows if r["cls"] == "neuron"])
    edges = morphology.dilation(segmentation.find_boundaries(labels, mode="inner"), morphology.disk(1))
    out[edges & neur] = (255, 204, 0)
    out[edges & glia] = (0, 212, 255)
    return out


def overlay(rgb, labels, rows, out_png):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].imshow(rgb); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(outlined(rgb, labels, rows))
    ax[1].set_title("Neurons (yellow)  ·  Glia (cyan)"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


if __name__ == "__main__":
    img = sys.argv[1]
    out = Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
    rgb, od, labels, rows, summary = segment(img)
    overlay(rgb, labels, rows, out / (Path(img).stem + "_overlay.png"))
    for k, v in summary.items():
        print(f"{k}: {v}")