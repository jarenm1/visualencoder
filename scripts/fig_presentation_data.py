"""Extract presentation images from the saved simulator, UAVid, and flight data.

From the repo root: `.venv/bin/python scripts/fig_presentation_data.py`.
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path('report/figures')
SIM = Path('data/train-v2')
MINED = Path('runs/mined-drone13/train')
PALETTE = np.array([[135, 190, 235], [110, 160, 70], [120, 120, 120],
                    [200, 80, 60], [30, 150, 40], [60, 80, 220], [230, 200, 40]], dtype=np.uint8)


def load(path, index):
    return np.load(path, mmap_mode='r')[index]


def save(name, *images):
    Image.fromarray(np.concatenate(images, axis=1)).save(OUT / name, optimize=True)


def main():
    pose = 92  # aligned residential block with materially different render styles
    looks = [load(SIM / f'rgb_random{i}.npy', pose) for i in (1, 2)]
    save('paired-training-views.png', *looks)
    save('uavid-training-view.png', load(Path('data/uavid/train_rgb.npy'), 0))
    frame = load(MINED / 'images.npy', 32).transpose(1, 2, 0)
    truth = PALETTE[load(SIM / 'semantic.npy', pose)]
    save('pilot-data-streams.png', looks[0], truth, frame)


if __name__ == '__main__':
    main()
