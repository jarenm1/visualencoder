"""Bounded offline test: direct visual memory readout with or without ego-motion transport."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from temporal_clips import generate
from temporal_pilot import BACKBONE, DEVICE, FPS, ROOT, amp, emit, extract
from visenc.models import load_encoder
from visenc.temporal import TemporalEncoder

LAYOUTS = {'train': ('train-101', 'train-102'), 'val': ('train-103',),
           'test': ('train-104', 'heldout-10002')}
GAPS = (0, FPS, 2 * FPS, 4 * FPS)
SCALE = 20.0


def body_axes(yaw):
    return np.stack((np.stack((-np.sin(yaw), -np.cos(yaw)), -1),
                     np.stack((np.cos(yaw), -np.sin(yaw)), -1)), -2)


def relative_motion(poses):
    axes = body_axes(np.array([pose.yaw for pose in poses]))
    pos = np.array([pose.position[[0, 2]] for pose in poses])
    return axes @ axes[0].T, np.einsum('tij,tj->ti', axes, pos - pos[0]) / SCALE


def collect_clips(backbone, train_count, eval_count, scene_seed, wide):
    enc = TemporalEncoder(backbone, 'inside').to(DEVICE).eval()
    splits = {}
    for i, (split, layouts) in enumerate(LAYOUTS.items()):
        clips = splits[split] = []
        for j, city in enumerate(layouts):
            count = train_count if split == 'train' else eval_count
            for rgb, target, h, meta in generate(ROOT / 'assets/geometry' / city, count,
                                                  scene_seed + 100 * i + j, sideways=True, wide=wide):
                assert np.array_equal(rgb[0, h:], rgb[1, h:]) and np.array_equal(rgb[0, h:], rgb[2, h:])
                x = extract(rgb, enc)
                n = h + max(GAPS) + 1
                # A stationary hold after the last rendered pose tests memory decay, not additional motion.
                time = np.minimum(np.arange(n), x.shape[1] - 1)
                rot, delta = relative_motion([meta['poses'][k] for k in time])
                bearing, distance = target[:, time].transpose(2, 0, 1)
                y = np.stack((np.cos(bearing) * distance, np.sin(bearing) * distance), -1) / SCALE
                oracle = np.einsum('tij,bj->bti', rot, y[:2, 0]) - delta
                np.testing.assert_allclose(oracle, y[:2], atol=2e-6, rtol=0)
                clips.append((x[:, time], torch.tensor(y), torch.tensor(rot, dtype=torch.float32),
                              torch.tensor(delta, dtype=torch.float32), h))
            emit(f'{split}/{city}: {count} triplets')
    return splits


def predict(enc, head, clip, use_motion, stop=None):
    x, y, rotation, displacement, _ = clip
    with amp():
        memory = enc.prefix_memory(x[:, :stop].to(DEVICE))
        raw = head(memory.flatten(2)).float()
    xy = raw[..., :2]
    if use_motion:
        xy = torch.einsum('tij,btj->bti', rotation[:stop].to(DEVICE), xy) - displacement[:stop].to(DEVICE)
    return xy, raw[..., 2], y[:, :stop].to(DEVICE)


def fit(backbone, clips, mode, use_motion, steps, seed):
    torch.manual_seed(seed)
    enc = TemporalEncoder(backbone, mode).to(DEVICE).train()
    # Same direct head, supervision, memory size, optimizer and trajectory draws for FIFO and recurrence.
    head = nn.Sequential(nn.LayerNorm(16 * enc.width), nn.Linear(16 * enc.width, 64), nn.GELU(),
                         nn.Linear(64, 3)).to(DEVICE)
    params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=6e-4)
    for k in np.random.default_rng(seed).integers(len(clips), size=steps):
        clip = clips[k]
        h = clip[-1]
        xy, confidence, y = predict(enc, head, clip, use_motion, h + 2 * FPS + 1)
        labels = torch.ones_like(confidence[:, 1:])
        labels[2] = 0
        loss = F.smooth_l1_loss(xy[:2, 1:], y[:2, 1:], beta=0.1)
        loss += 0.25 * F.binary_cross_entropy_with_logits(confidence[:, 1:], labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
    return enc.eval(), head.eval()


@torch.no_grad()
def records(enc, head, clips, motion):
    rows = []
    for clip in clips:
        xy, confidence, y = predict(enc, head, clip, motion)
        times = [clip[-1] + g for g in GAPS]
        for_predict = xy[:2, times]
        answer = y[:2, times]
        delta = torch.atan2(for_predict[..., 1], for_predict[..., 0]) - torch.atan2(answer[..., 1], answer[..., 0])
        deg = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs())
        range_m = (for_predict.norm(dim=-1) - answer.norm(dim=-1)).abs() * SCALE
        rows.append((deg.cpu().numpy(), range_m.cpu().numpy(), confidence[:, times].cpu().numpy()))
    return tuple(np.stack(col) for col in zip(*rows))


def pose_only(clips):
    """Predict the typical reveal-frame point for every clip, without reading any pixels."""
    rows = {}
    for gap in GAPS:
        measures = []
        for _, y, rot, delta, h in clips:
            t = h + gap
            xy = rot[t] @ torch.tensor([15. / SCALE, 0.]) - delta[t]
            deg = torch.rad2deg((torch.atan2(y[:2, t, 1], y[:2, t, 0]) - torch.atan2(xy[1], xy[0])).abs())
            distance = SCALE * (y[:2, t].norm(dim=-1) - xy.norm()).abs()
            measures.append(torch.stack((deg, distance)))
        angle, distance = torch.cat(measures, dim=1)
        rows[f'{gap / FPS:g}s'] = {'unthresholded': float(((angle <= 10) & (distance <= 5)).float().mean()),
                                  'bearing_mae_deg': float(angle.mean()), 'range_mae_m': float(distance.mean())}
    return rows


def metrics(validation, test, seed):
    angle, distance, confidence = test
    # Calibrate once per variant using all sampled gaps on each absent validation path.
    threshold = float(np.quantile(validation[2][:, 2].max(axis=1), .95, method='higher'))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(angle), size=(2000, len(angle)))
    rows = {}
    for i, gap in enumerate(GAPS):
        plausible = (angle[:, :, i] <= 10) & (distance[:, :, i] <= 5)
        hits = plausible & (confidence[:, :2, i] > threshold)
        gap_threshold = np.quantile(validation[2][:, 2, i], .95, method='higher')
        pair = confidence[:, :2, i].reshape(-1, 1) - confidence[:, 2, i]
        rows[f'{gap / FPS:g}s'] = {
            'recall': float(hits.mean()), 'recall_ci95': np.percentile(hits.mean(axis=1)[draws].mean(axis=1), [2.5, 97.5]).tolist(),
            'unthresholded': float(plausible.mean()), 'false_memory_paths': int((confidence[:, 2, i] > threshold).sum()),
            'gap_calibrated_recall': float((plausible & (confidence[:, :2, i] > gap_threshold)).mean()),
            'gap_calibrated_false_paths': int((confidence[:, 2, i] > gap_threshold).sum()),
            'presence_auroc': float(((pair > 0) + 0.5 * (pair == 0)).mean()),
            'bearing_mae_deg': float(angle[:, :, i].mean()), 'range_mae_m': float(distance[:, :, i].mean()),
            'present_above_absent': float(((confidence[:, :2, i] > confidence[:, 2:3, i]) +
                                           .5 * (confidence[:, :2, i] == confidence[:, 2:3, i])).mean()),
            'recall_by_path': hits.mean(axis=1).tolist(),
        }
    return {'threshold': threshold, 'paths': len(angle), 'gaps': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--train-per-city', type=int, default=10)
    parser.add_argument('--eval-per-city', type=int, default=12)
    parser.add_argument('--scene-seed', type=int, default=2701)
    parser.add_argument('--transport-only', action='store_true')
    parser.add_argument('--wide', action='store_true', help='vary reveal distance and lateral offset beyond the original task')
    args = parser.parse_args()
    if not (1 <= args.steps <= 150 and 1 <= args.train_per_city <= 12 and 2 <= args.eval_per_city <= 16):
        parser.error('caps: 150 steps, 12 training and 16 evaluation paths per city')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.set_float32_matmul_precision('high')
    backbone = load_encoder(BACKBONE).to(DEVICE).eval().requires_grad_(False)
    data = collect_clips(backbone, args.train_per_city, args.eval_per_city, args.scene_seed, args.wide)
    prior = pose_only(data['test'])
    emit(f'Pose-only prior: {prior}')
    rows = {}
    for seed in (17, 29):
        for mode in ('fifo', 'inside'):
            for motion in ((True,) if args.transport_only else (False, True)):
                enc, head = fit(backbone, data['train'], mode, motion, args.steps, seed)
                val, test = (records(enc, head, data[s], motion) for s in ('val', 'test'))
                key = f'{mode}_{"transport" if motion else "rgb"}'
                row = metrics(val, test, seed)
                rows.setdefault(key, {})[str(seed)] = row
                emit(f'{seed} {key}: ' + ', '.join(f'{g} {r["recall"]:.1%} / {r["bearing_mae_deg"]:.1f} deg'
                                                  for g, r in row['gaps'].items()))
                result = {'config': vars(args), 'layouts': LAYOUTS, 'variants': rows, 'pose_only': prior,
                          'peak_torch_cuda_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}
                (args.out / 'geometry.json').write_text(json.dumps(result, default=str, indent=2) + '\n')


if __name__ == '__main__':
    main()
