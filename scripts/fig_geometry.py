"""Representative layouts and paired appearances, chosen without model outputs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    manifest = json.loads(Path("data/geometry/geometry.json").read_text())
    entries = [manifest["arms"]["fixed"]["familiar"][0],
               *manifest["arms"]["multi"]["familiar"][1:], *manifest["heldout"]]
    labels = ["Original training city", "Training layout 101", "Training layout 102",
              "Held-out layout 10001", "Held-out layout 10002"]
    width, height = manifest["config"]["width"], manifest["config"]["height"]
    sheet = Image.new("RGB", (len(entries) * width, 2 * (height + 25)))
    draw = ImageDraw.Draw(sheet)
    selected = []
    for column, (entry, label) in enumerate(zip(entries, labels)):
        split = Path(entry["split"])
        semantic = np.load(split / "semantic.npy", mmap_mode="r")[:entry["count"]]
        state = np.load(split / "state.npy", mmap_mode="r")[:entry["count"]]
        # Select a mixed urban view, not the best/worst encoder prediction.
        building = (semantic == 3).mean((1, 2))
        road = (semantic == 2).mean((1, 2))
        vegetation = (semantic == 4).mean((1, 2))
        score = abs(building - 0.35) + abs(road - 0.2) + abs(vegetation - 0.2)
        score += ((state[:, 1] < 10) | (state[:, 1] > 30)) * 2
        index = int(score.argmin())
        selected.append({"split": str(split), "index": index})
        for row, view in enumerate(("canonical", "novel1")):
            pixels = np.load(split / f"rgb_{view}.npy", mmap_mode="r")[index]
            y = row * (height + 25)
            draw.text((column * width + 6, y + 5), f"{label}: {view}", fill="white")
            sheet.paste(Image.fromarray(pixels), (column * width, y + 25))
    sheet.save("report/figures/geometry.png")
    out = Path("reports/experiments/geometry/figure.json")
    out.write_text(json.dumps({"selection": "Mixed urban ground-truth coverage and camera height; no encoder outputs",
                               "selected": selected}, indent=2) + "\n")
    print("report/figures/geometry.png")


if __name__ == "__main__":
    main()
