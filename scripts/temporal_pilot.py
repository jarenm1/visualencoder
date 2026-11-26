"""Run the offline temporal encoder pilot (no PPO or external downloads)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from temporal_clips import generate
from visenc import real
from visenc.explore import Encoder as ProbeEncoder
from visenc.models import load_encoder
from visenc.temporal import TemporalEncoder

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = 'run:runs/refine-anchors-mv'
SPLITS = (('train', ('train-101', 'train-102')), ('val', ('heldout-10001',)),
          ('test', ('heldout-10002',)))
FPS = 4
GRID = (15, 20)
DEVICE = 'cuda'


def emit(message):
    print(message, flush=True)


def tensor(x):
    return torch.as_tensor(x, device=DEVICE)


def images(rgb):
    return tensor(rgb).permute(0, 3, 1, 2).float().div_(255)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


@torch.no_grad()
def extract(rgb, adapter):
    # Only one encode per distinct RGB frame within the matched triplet.
    index = {}
    unique, ids = [], []
    for frame in rgb.reshape(-1, *rgb.shape[2:]):
        key = frame.tobytes()
        if key not in index:
            index[key] = len(unique)
            unique.append(frame)
        ids.append(index[key])
    encoded = []
    for start in range(0, len(unique), 4):
        with amp():
            encoded.append(adapter.prefix(images(np.stack(unique[start:start + 4]))).to('cpu', torch.float16))
    return torch.cat(encoded)[ids].unflatten(0, rgb.shape[:2])


def collect(args, backbone):
    adapter = TemporalEncoder(backbone, 'inside').to(DEVICE).eval()
    data = {}
    for i, (split, cities) in enumerate(SPLITS):
        data[split] = []
        for j, city in enumerate(cities):
            count = args.train_per_city if split == 'train' else args.eval_per_city
            for clip in generate(ROOT / 'assets/geometry' / city, count, args.seed + 1000 * i + 100 * j):
                rgb, target, hidden, *extra = clip
                assert hidden >= 1 and hidden + 2 * FPS < rgb.shape[1], (hidden, rgb.shape)
                assert np.all(rgb[:, hidden:] == rgb[0, hidden:]), 'RGB leakage in scored suffix'
                data[split].append((extract(rgb, adapter), target, hidden, extra[0] if extra else {}))
                if args.map_checkpoint:
                    # Keep only the held-out RGB for comparison with the existing map stack.
                    if split != 'train':
                        data[split][-1][3]['rgb'] = rgb
            emit(f'{split} {city}: {count} matched triplets')
    return data


def train(mode, backbone, clips, steps, seed, *, side=False):
    torch.manual_seed(seed)
    adapter = TemporalEncoder(backbone, mode).to(DEVICE).train()
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=8e-4)
    rng = np.random.default_rng(seed)
    assert not any(p.requires_grad for p in backbone.parameters())
    for step in range(steps):
        x, y, hidden, _ = clips[int(rng.integers(len(clips)))]
        if side:
            # Reveal-side classification, not the sign of bearing after the camera turns.
            with amp():
                pred = adapter.forward_cached(x[:2, :hidden + 1].to(DEVICE, dtype=torch.float32), grid=GRID)
            loss = F.binary_cross_entropy_with_logits(pred[:, -1, 0].float(), tensor([0., 1.]))
        else:
            end = hidden + 2 * FPS + 1
            with amp():
                pred = adapter.forward_cached(x[:, :end].to(DEVICE, dtype=torch.float32), grid=GRID).float()
            y = tensor(y[:, :end])
            logit = pred[:, hidden:, 2]
            present = torch.ones_like(logit)
            present[2] = 0
            angle = pred[:2, hidden:, 0] - y[:2, hidden:, 0]
            location = (1 - angle.cos()).mean() + F.smooth_l1_loss(pred[:2, hidden:, 1], y[:2, hidden:, 1] / 20)
            loss = location + 0.5 * F.binary_cross_entropy_with_logits(logit, present)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            emit(f'{mode} {step + 1}/{steps}, loss {loss.item():.3f}')
    return adapter.eval()


@torch.no_grad()
def predict(model, clips):
    predictions = []
    for x, y, h, _ in clips:
        with amp():
            pred = model.forward_cached(x.to(DEVICE, dtype=torch.float32), grid=GRID).float().cpu().numpy()
        start, end = h + FPS, h + 2 * FPS + 1
        predictions.append((pred[:, start:end], y[:, start:end]))
    return predictions


def eligible(pred, target):
    delta = pred[..., 0] - target[..., 0]
    error = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    return (error <= math.radians(10)) & (np.abs(pred[..., 1] * 20 - target[..., 1]) <= 5)


def score(val, test):
    # One clip is the calibration unit: any positive memory during a negative clip counts.
    negative = np.stack([p[2, :, 2].max() for p, _ in val])
    threshold = float(np.quantile(negative, 0.95, method='higher'))
    hit = np.stack([eligible(p[:2], y[:2]) & (p[:2, :, 2] > threshold) for p, y in test])
    false = [p[2, :, 2].max() > threshold for p, _ in test]
    mae = []
    for p, y in test:
        delta = p[:2, :, 0] - y[:2, :, 0]
        mae.append(np.abs(np.arctan2(np.sin(delta), np.cos(delta))).mean())
    eligible_positives = [eligible(p[:2], y[:2]).mean() for p, y in test]
    confidence_gap = [p[:2, :, 2].mean() - p[2, :, 2].mean() for p, _ in test]
    return {'threshold': threshold, 'recall': float(hit.mean()),
            'unthresholded_localization': float(np.mean(eligible_positives)),
            'mean_present_minus_never_logit_or_evidence': float(np.mean(confidence_gap)),
            'false_memory_clips': int(sum(false)), 'test_negative_clips': len(test),
            'bearing_mae_deg': float(np.degrees(np.mean(mae))), 'trajectory_recall': hit.mean(axis=(1, 2)).tolist()}


@torch.no_grad()
def map_predictions(checkpoint, clips):
    # This is the existing deployed map stack, not a parameter-matched visual ablation.
    from PIL import Image

    sys.path.insert(0, str(ROOT / 'sim/python'))
    from sim_rl.memory import EGO_RADIUS, RATE_SCALE, VELOCITY_SCALE, SpatialMemory
    from sim_rl.model import ActorCritic, goal_tokens
    from sim_rl.native import CONTROL_HZ
    from sim_rl.perception import HEIGHT_SCALE, ground_prior

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    assert meta['goal'] == 'patch' and meta['memory']
    dim = meta['feature_dim']
    policy = ActorCritic('visual', dim, dim, 15, 20, meta['action_dim'], 'patch', True,
                         meta['commands'], meta['memory']['depth_view'], meta.get('teacher_target', False))
    policy.load_state_dict(saved['model'])
    policy.to(DEVICE).eval()
    encoder = load_encoder(meta['encoder']).to(DEVICE).eval()
    predictions = []
    goals = {}
    for _, y, h, info in clips:
        poses = info['poses']
        goal_name = info['car']
        if goal_name not in goals:
            with Image.open(ROOT / 'runs/reference-library' / f'cars-{goal_name}.png') as image:
                goal_rgb = np.asarray(image.convert('RGB').resize((320, 240))).copy()
            with amp():
                cls, patch = encoder(images(goal_rgb[None]))
            goals[goal_name] = goal_tokens(cls, patch, 15, 20).expand(3, -1, -1)
        memory = SpatialMemory(3, 15, 20, {'aspect': 4/3, 'uptilt_rad': poses[0].pitch,
                                           'vertical_fov_rad': poses[0].vfov}, torch.device(DEVICE))
        trace = []
        for t, pose in enumerate(poses):
            assert pose.pitch == poses[0].pitch and pose.roll == 0 and pose.vfov == poses[0].vfov
            last = poses[max(t - 1, 0)]
            motion = pose.position - last.position
            forward = np.array([-math.sin(pose.yaw), 0., -math.cos(pose.yaw)])
            right = np.cross(forward, [0., 1., 0.])
            proprio = torch.zeros((3, 19), device=DEVICE)
            proprio[:, 1] = -1
            proprio[:, 4] = (pose.yaw - last.yaw) * CONTROL_HZ / RATE_SCALE
            proprio[:, 6] = float(motion @ right) * CONTROL_HZ / VELOCITY_SCALE
            proprio[:, 8] = -float(motion @ forward) * CONTROL_HZ / VELOCITY_SCALE
            proprio[:, 9] = float(pose.position[1]) / HEIGHT_SCALE
            with amp():
                cls, patch = encoder(images(info['rgb'][:, t]))
            _, logits = policy.attention(goals[goal_name], cls, patch)
            evidence = (logits[:, :-1] - logits[:, -1:]).sigmoid()
            depth, _ = policy.perception.predict(patch, ground_prior(proprio, memory.rays))
            memory.update(proprio, depth, evidence)
            candidate = memory.candidate().cpu().numpy()
            trace.append(np.stack([np.arctan2(candidate[:, 1], candidate[:, 0]),
                                   candidate[:, 2] * EGO_RADIUS / 20, candidate[:, 3]], axis=-1))
        sl = slice(h + FPS, h + 2 * FPS + 1)
        predictions.append((np.stack(trace, axis=1)[:, sl], y[:, sl]))
    return predictions


@torch.no_grad()
def check_and_time(model, clip):
    x, _, h, _ = clip
    x = x.to(DEVICE, dtype=torch.float32)
    with amp():
        pred = model.forward_cached(x, grid=GRID).float()
        prefix_pred = model.forward_cached(x[:, :h + 1], grid=GRID).float()
        torch.testing.assert_close(pred[:, :h + 1], prefix_pred, atol=0.02, rtol=0.02)
        # With identical suffix pixels, swapping only history must swap the predictions.
        swapped = torch.cat((x[[1, 0, 2], :h], x[:, h:]), dim=1)
        other = model.forward_cached(swapped, grid=GRID).float()
        torch.testing.assert_close(other[:, h:], pred[[1, 0, 2], h:], atol=0.02, rtol=0.02)
        delta = pred[0, h + FPS:h + 2 * FPS + 1, 0] - other[0, h + FPS:h + 2 * FPS + 1, 0]
        history_deg = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs().mean()).item()
    return history_deg


@torch.no_grad()
def latency(model, temporal):
    pic = torch.zeros((1, 3, 240, 320), device=DEVICE)
    state, elapsed = None, []
    with amp():
        for i in range(30):
            torch.cuda.synchronize()
            before = time.perf_counter()
            if temporal:
                _, _, _, state = model.step_cached(model.prefix(pic), state, grid=GRID)
            else:
                model(pic)
            torch.cuda.synchronize()
            if i >= 10:
                elapsed.append(1000 * (time.perf_counter() - before))
    return np.percentile(elapsed, [50, 95]).tolist()


class LoadedProbeEncoder(ProbeEncoder):
    def __init__(self, spec, model):
        self.spec, self.device, self.batch_size = spec, torch.device(DEVICE), 4
        self.model = model


def real_check(out, baseline, model, fit):
    encoders = {'frozen': baseline, 'temporal-inside-reset': model}
    real.run(SimpleNamespace(sim='data/eval-v2', sim_view='canonical', sim_fit=fit, real='data/uavid',
                             encoders=list(encoders), out=out, baseline='frozen', bootstrap_resamples=200),
             encoder_factory=lambda spec, **_: LoadedProbeEncoder(spec, encoders[spec]))


@torch.no_grad()
def side_logits(model, clips):
    rows = []
    for x, _, hidden, _ in clips:
        x = x[:2, :hidden + 1].to(DEVICE, dtype=torch.float32)
        swapped = torch.cat((x.flip(0)[:, :hidden], x[:, hidden:]), dim=1)
        with amp():
            logits = model.forward_cached(x, grid=GRID)[:, -1, 0].float()
            other = model.forward_cached(swapped, grid=GRID)[:, -1, 0].float()
        torch.testing.assert_close(other, logits.flip(0), rtol=0, atol=0.01)
        rows.append(logits.cpu().numpy())
    return np.stack(rows)


def side_summary(rows, draws):
    right_probability = 1 / (1 + np.exp(-rows))
    correct = (rows[:, 0] <= 0).astype(float) / 2 + (rows[:, 1] > 0) / 2
    gap = rows[:, 1] - rows[:, 0]
    return {
        'accuracy': float(correct.mean()),
        'accuracy_ci95': np.percentile(correct[draws].mean(axis=1), [2.5, 97.5]).tolist(),
        'both_correct': int(((rows[:, 0] <= 0) & (rows[:, 1] > 0)).sum()),
        'paired_right_rank': float(((gap > 0) + 0.5 * (gap == 0)).mean()),
        'paired_right_rank_ci95': np.percentile(
            ((gap > 0) + 0.5 * (gap == 0))[draws].mean(axis=1), [2.5, 97.5]).tolist(),
        'pair_ties': int((gap == 0).sum()),
        'mean_absolute_swap_probability_delta': float(np.abs(np.diff(right_probability, axis=1)).mean()),
        'logits_left_right': rows.tolist(),
    }


@torch.no_grad()
def probe_features(model, clips, *, visible, current=False):
    rows = []
    for x, _, hidden, _ in clips:
        with amp():
            if visible or current:
                t = hidden if current else 0
                _, patches = model.cold(x[:2, t:t + 1].to(DEVICE), grid=GRID)
                grid = patches[:, 0].float().transpose(1, 2).reshape(2, -1, *GRID)
                features = F.adaptive_avg_pool2d(grid, (3, 4)).flatten(1)
            else:
                state = None
                for frame in x[:2, :hidden].to(DEVICE).unbind(dim=1):
                    _, _, _, state = model.step_cached(frame, state, grid=GRID)
                # Incoming state at the first pixel-identical frame; no contribution from that frame.
                features = state[0].flatten(1)
        rows.append(features.float().cpu().numpy())
    return np.stack(rows)


def fit_side_probe(features, draws):
    # Learn on within-path contrasts to remove shared scenery without exposing test labels.
    train = features['train'].astype(np.float64)
    center = train.mean(axis=(0, 1))
    differences = train[:, 1] - train[:, 0]
    gram = differences @ differences.T
    penalty = max(0.1 * np.trace(gram) / len(gram), 1e-12)
    weights = differences.T @ np.linalg.solve(gram + penalty * np.eye(len(gram)), np.ones(len(gram)))
    result = {}
    for split, x in features.items():
        summary = side_summary((x.astype(np.float64) - center) @ weights, draws[split])
        del summary['mean_absolute_swap_probability_delta']  # Ridge scores are not calibrated logits.
        summary['scores_left_right'] = summary.pop('logits_left_right')
        result[split] = summary
    return result


def fit_clip_readout(blocks, draws):
    """Fit a linear classifier on individual clips, with equally scaled input blocks."""
    normalized = []
    for block in blocks:
        train = block['train'].astype(np.float64)
        center = train.mean(axis=(0, 1), keepdims=True)
        scale = max(np.sqrt(np.square(train - center).sum(axis=-1).mean()), 1e-12)
        normalized.append({split: (x.astype(np.float64) - center) / scale for split, x in block.items()})
    inputs = {split: np.concatenate([b[split] for b in normalized], axis=-1) for split in draws}
    train = inputs['train'].reshape(-1, inputs['train'].shape[-1])
    labels = np.tile([-1., 1.], len(inputs['train']))
    gram = train @ train.T
    penalty = 0.1 * np.trace(gram) / len(gram)
    weights = train.T @ np.linalg.solve(gram + max(penalty, 1e-12) * np.eye(len(gram)), labels)
    result = {}
    for split, x in inputs.items():
        scores = x @ weights
        # Replace only the history block; the current image is unchanged.
        if len(normalized) > 1:
            frame, history = (b[split] for b in normalized)
            swapped = np.concatenate((frame, history[:, ::-1]), axis=-1) @ weights
            np.testing.assert_allclose(swapped, scores[:, ::-1], atol=1e-9, rtol=0)
        summary = side_summary(scores, draws[split])
        del summary['mean_absolute_swap_probability_delta']
        summary['scores_left_right'] = summary.pop('logits_left_right')
        summary['decision_flip_rate_on_swap'] = float(((scores[:, 0] > 0) != (scores[:, 1] > 0)).mean())
        result[split] = summary
    return result


def run_side(args, backbone, data):
    rng = np.random.default_rng(args.seed)
    draws = {split: rng.integers(len(clips), size=(2000, len(clips))) for split, clips in data.items()}
    results = {'config': vars(args), 'variants': {}}
    if args.probe or args.readout:
        cold = TemporalEncoder(backbone, 'inside').to(DEVICE).eval()
        if args.probe:
            results['probes'] = {
                'visible_patches': fit_side_probe(
                    {split: probe_features(cold, clips, visible=True) for split, clips in data.items()}, draws)
            }
        if args.readout:
            frame = {split: probe_features(cold, clips, visible=False, current=True)
                     for split, clips in data.items()}
            results['readouts'] = {'frame_only': fit_clip_readout([frame], draws)}
        del cold
    for mode in ('reset', 'fifo', 'inside', 'late'):
        model = train(mode, backbone, data['train'], args.steps, args.seed, side=True)
        row = {split: side_summary(side_logits(model, clips), draws[split]) for split, clips in data.items()}
        results['variants'][mode] = row
        if (args.probe or args.readout) and mode != 'reset':
            slots = {split: probe_features(model, clips, visible=False) for split, clips in data.items()}
            if args.probe:
                results['probes'][f'{mode}_state'] = fit_side_probe(slots, draws)
            if args.readout:
                results['readouts'][f'{mode}_state_only'] = fit_clip_readout([slots], draws)
                results['readouts'][f'{mode}_state_and_frame'] = fit_clip_readout([frame, slots], draws)
        emit(f"{mode}: train/val/test accuracy = "
             + '/'.join(f"{row[split]['accuracy']:.1%}" for split in ('train', 'val', 'test'))
             + f"; test pair rank {row['test']['paired_right_rank']:.1%}, "
             + f"swap delta {row['test']['mean_absolute_swap_probability_delta']:.4f}")
    results['peak_torch_cuda_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    (args.out / 'side.json').write_text(json.dumps(results, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-per-city', type=int, default=12)
    parser.add_argument('--eval-per-city', type=int, default=12)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--sim-fit', type=int, default=120, help='bounded UAVid sim-probe fit poses')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--probe', action='store_true', help='with --side, probe frozen visible patches and incoming memory state')
    parser.add_argument('--readout', action='store_true', help='with --side, compare individual-clip frozen state/frame readouts')
    parser.add_argument('--side', action='store_true', help='classify reveal side at the first identical frame; skip map and UAVid')
    parser.add_argument('--map-checkpoint', type=Path, default=Path('runs/drone13-v5-seed0/best.pt'))
    args = parser.parse_args()
    if (args.probe or args.readout) and not args.side:
        parser.error('--probe and --readout require --side')
    if not (1 <= args.train_per_city <= 24 and 2 <= args.eval_per_city <= 24 and 1 <= args.steps <= 200):
        parser.error('pilot caps: train/city 1–24, eval/city 2–24, steps 1–200')
    if not 16 <= args.sim_fit <= 200:
        parser.error('pilot caps: sim-fit 16–200')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.20)
    torch.set_float32_matmul_precision('high')
    backbone = load_encoder(BACKBONE).to(DEVICE).eval().requires_grad_(False)
    if args.side:
        args.map_checkpoint = None
    data = collect(args, backbone)
    if args.side:
        run_side(args, backbone, data)
        return
    results = {'config': vars(args).copy(), 'variants': {}, 'frozen_step_p50_p95_ms': latency(backbone, False)}
    models = {}
    for mode in ('reset', 'fifo', 'inside', 'late'):
        model = train(mode, backbone, data['train'], args.steps, args.seed)
        row = score(predict(model, data['val']), predict(model, data['test']))
        row['counterfactual_bearing_shift_deg'] = check_and_time(model, data['test'][0])
        row['step_p50_p95_ms'] = latency(model, True)
        models[mode] = model
        results['variants'][mode] = row
        emit(f'{mode}: {row}')
        (args.out / 'pilot.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    rng = np.random.default_rng(args.seed)
    for baseline in ('reset', 'fifo', 'late'):
        a = np.asarray(results['variants']['inside']['trajectory_recall'])
        b = np.asarray(results['variants'][baseline]['trajectory_recall'])
        draws = rng.integers(len(a), size=(1000, len(a)))
        results['variants']['inside'][f'paired_ci95_vs_{baseline}'] = np.percentile((a - b)[draws].mean(1), [2.5, 97.5]).tolist()
    if args.map_checkpoint:
        row = score(map_predictions(args.map_checkpoint, data['val']), map_predictions(args.map_checkpoint, data['test']))
        results['variants']['current_map'] = row
        emit(f'current_map: {row}')
    (args.out / 'pilot.json').write_text(json.dumps(results, indent=2, default=str) + '\n')
    model = models['inside']
    torch.save({k: v for k, v in model.state_dict().items() if k in dict(model.named_parameters())
                and dict(model.named_parameters())[k].requires_grad}, args.out / 'inside-adapter.pt')
    real_check(args.out / 'uavid-cold', backbone, model, args.sim_fit)
    results['peak_torch_cuda_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    (args.out / 'pilot.json').write_text(json.dumps(results, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
