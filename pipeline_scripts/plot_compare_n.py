"""Put each trip's HMM track plots for every n side by side, so the effect of n on the same trip is visible.

Input:  result_figures/hmm_tracks/n_<k>/<trip>.png (from plot_hmm_tracks.py)
Output: result_figures/hmm_tracks/compare_n/<trip>.png, one row of panels labelled "n = k"

Usage: /usr/bin/python3 pipeline_scripts/plot_compare_n.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = PROJECT_ROOT / "result_figures" / "hmm_tracks"
OUT_DIR = TRACKS_DIR / "compare_n"
LABEL_HEIGHT = 70          # white band above each panel for the "n = k" label
GAP = 20                   # white space between panels


def main() -> None:
    n_dirs = sorted((d for d in TRACKS_DIR.glob("n_*") if d.is_dir()), key=lambda d: int(d.name.split("_")[1]))
    trips = sorted({p.name for d in n_dirs for p in d.glob("*.png")})
    OUT_DIR.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except OSError:
        font = ImageFont.load_default()

    for trip in trips:
        panels = [(d.name.split("_")[1], Image.open(d / trip).convert("RGB")) for d in n_dirs if (d / trip).exists()]
        height = max(img.height for _, img in panels)
        width = sum(img.width for _, img in panels) + GAP * (len(panels) - 1)
        sheet = Image.new("RGB", (width, height + LABEL_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)
        x = 0
        for n, img in panels:
            sheet.paste(img, (x, LABEL_HEIGHT))
            label = f"n = {n}"
            text_w = draw.textlength(label, font=font)
            draw.text((x + (img.width - text_w) / 2, 10), label, fill="black", font=font)
            x += img.width + GAP
        sheet.save(OUT_DIR / trip)
        print(f"Saved {OUT_DIR / trip} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
