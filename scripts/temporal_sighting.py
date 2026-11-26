"""Localize a car on one frozen frame, then retain that observation without recurrent state."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

from temporal_geometry import GAPS, SCALE, collect_clips, metrics
from temporal_pilot import BACKBONE, DEVICE, FPS, GRID, amp
from visenc.models import load_encoder
from visenc.temporal import TemporalEncoder


@torch.no_grad()
def features(encoder, clips):
    cls, patches = [], []
    for x, *_ in clips:
        with amp():
            global_token, patch_tokens = encoder.cold(x[:, :1].to(DEVICE), grid=GRID)
            grid = patch_tokens[:, 0].float().transpose(1, 2).reshape(3, -1, *GRID)
            spatial = adaptive_avg_pool2d(grid, (3, 4)).flatten(1)
        cls.append(global_token[:, 0].float().cpu().numpy())
        patches.append(spatial.cpu().numpy())
    return {'cls': np.stack(cls), 'patches': np.stack(patches)}


def ridge(x, y):
    center = x.mean(0)
    target_mean = y.mean(0)
    centered = x - center
    kernel = centered @ centered.T
    penalty = max(.05 * np.trace(kernel) / len(x), 1e-12)
    weight = centered.T @ np.linalg.solve(kernel + penalty * np.eye(len(x)), y - target_mean)
    return lambda z: (z - center) @ weight + target_mean


def fit(blocks, targets):
    # Negatives supervise presence only. No synthetic look or paired RGB difference is a model input.
    x = blocks.astype(np.float64)
    location = ridge(x[:, :2].reshape(-1, x.shape[-1]), targets.reshape(-1, 2))
    confidence = ridge(x.reshape(-1, x.shape[-1]), np.tile([1., 1., -1.], len(x)))
    return lambda z: (location(z.astype(np.float64)), confidence(z.astype(np.float64)))


def records(predictions, clips, *, reveal=False, drift_yaw_deg_s=0., drift_right_m_s=0.):
    positions, scores = predictions
    angles, distances = [], []
    for (xy, (_, y, rotation, displacement, hidden)) in zip(positions, clips, strict=True):
        # The reveal estimate and its confidence are fixed; later RGB is never consumed.
        idx = [0] * len(GAPS) if reveal else [hidden + gap for gap in GAPS]
        xy = np.einsum('tij,bj->bti', rotation[idx].numpy(), xy[:2]) - displacement[idx].numpy()
        elapsed = np.asarray(idx) / FPS
        yaw = np.deg2rad(drift_yaw_deg_s * elapsed)
        bias = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).transpose(2, 0, 1)
        xy = np.einsum('tij,btj->bti', bias, xy)
        xy[..., 1] -= drift_right_m_s * elapsed / SCALE
        truth = y[:2, idx].numpy()
        angle = np.arctan2(xy[..., 1], xy[..., 0]) - np.arctan2(truth[..., 1], truth[..., 0])
        angles.append(np.abs(np.rad2deg(np.arctan2(np.sin(angle), np.cos(angle)))))
        distances.append(SCALE * np.abs(np.linalg.norm(xy, axis=-1) - np.linalg.norm(truth, axis=-1)))
    return np.stack(angles), np.stack(distances), np.repeat(scores[:, :, None], len(GAPS), axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene-seed', type=int, default=5831)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.set_float32_matmul_precision('high')
    backbone = load_encoder(BACKBONE).to(DEVICE).eval().requires_grad_(False)
    data = collect_clips(backbone, 10, 12, args.scene_seed, wide=True)
    encoder = TemporalEncoder(backbone, 'inside').to(DEVICE).eval()
    feats = {split: features(encoder, clips) for split, clips in data.items()}
    target = np.stack([clip[1][:2, 0].numpy() for clip in data['train']])
    result = {}
    for kind in ('cls', 'patches'):
        readout = fit(feats['train'][kind], target)
        val, test = (readout(feats[key][kind]) for key in ('val', 'test'))
        calibration = records(val, data['val'], reveal=True)
        result[kind] = {
            'reveal': metrics(calibration, records(test, data['test'], reveal=True), 17)['gaps']['0s'],
            'exact_motion': metrics(calibration, records(test, data['test']), 17),
        }
        for sign in (-1, 1):
            perturbed = records(test, data['test'], drift_yaw_deg_s=sign * 2., drift_right_m_s=sign * .25)
            result[kind][f'drift_{sign:+d}'] = metrics(calibration, perturbed, 17)
        reveal = result[kind]['reveal']
        exact = result[kind]['exact_motion']['gaps']
        print(f'{kind}: reveal recall {reveal["recall"]:.1%}, '
              + ', '.join(f'{gap} recall {r["recall"]:.1%}, bearing {r["bearing_mae_deg"]:.1f}°, '
                          f'range {r["range_mae_m"]:.1f} m' for gap, r in exact.items()), flush=True)
    output = {'scene_seed': args.scene_seed, 'variants': result,
              'peak_torch_cuda_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    (args.out / 'sighting.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
