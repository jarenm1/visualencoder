"""Test whether slot supervision and a nonzero reader gate rescue causal visual memory."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from temporal_pilot import BACKBONE, DEVICE, GRID, amp, collect, emit, side_summary
from visenc.models import load_encoder
from visenc.temporal import TemporalEncoder

GAPS = (0, 2, 4, 8)  # frames after first identical frame; video is sampled at 4 Hz
VARIANTS = (
    ('fifo', 'fifo', False, 0.),
    ('inside', 'inside', False, 0.),
    ('inside_gate', 'inside', False, 0.05),
    ('inside_aux', 'inside', True, 0.),
    ('inside_aux_gate', 'inside', True, 0.05),
    ('late_aux_gate', 'late', True, 0.05),
)


def clip_input(clip):
    x, _, h, _ = clip
    return x[:2, :h + GAPS[-1] + 1].to(DEVICE, dtype=torch.float32), h


def fit(backbone, clips, steps, seed, mode, supervised, gate):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalEncoder(backbone, mode).to(DEVICE).train()
    nn.init.constant_(model.reader.gate, gate)
    slot_head = (nn.Sequential(nn.LayerNorm(16 * model.width), nn.Linear(16 * model.width, 32),
                               nn.GELU(), nn.Linear(32, 1)).to(DEVICE) if supervised else None)
    params = [p for p in model.parameters() if p.requires_grad]
    if slot_head is not None:
        params.extend(slot_head.parameters())
    opt = torch.optim.AdamW(params, lr=8e-4)
    labels = torch.tensor([0., 1.], device=DEVICE)[:, None]
    for i in rng.integers(len(clips), size=steps):
        x, h = clip_input(clips[i])
        with amp():
            slot_loss = None
            if slot_head is None:
                pred = model.forward_cached(x, grid=GRID)
            else:
                pred, slots = model.supervised_cached(x, grid=GRID)
                slot_logits = slot_head(slots[:, 1:].flatten(2)).squeeze(-1).float()
                slot_loss = F.binary_cross_entropy_with_logits(slot_logits, labels.expand_as(slot_logits))
            logits = pred[:, [h + gap for gap in GAPS], 0].float()
            loss = F.binary_cross_entropy_with_logits(logits, labels.expand_as(logits))
            if slot_loss is not None:
                loss = loss + slot_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
    return model.eval(), slot_head.eval() if slot_head is not None else None


@torch.no_grad()
def measure(model, slot_head, clips, draws):
    rows, auxiliary = [], []
    for clip in clips:
        x, h = clip_input(clip)
        idx = [h + gap for gap in GAPS]
        with amp():
            if slot_head is None:
                pred = model.forward_cached(x, grid=GRID)
            else:
                pred, slots = model.supervised_cached(x, grid=GRID)
                auxiliary.append(slot_head(slots[:, idx].flatten(2)).squeeze(-1).float().cpu().numpy())
        rows.append(pred[:, idx, 0].float().cpu().numpy())
    result = {}
    for name, values in (('visual_head', rows), ('slot_head', auxiliary)):
        if values:
            scores = np.stack(values)
            result[name] = {f'{gap / 4:g}s': side_summary(scores[:, :, j], draws)
                            for j, gap in enumerate(GAPS)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--train-per-city', type=int, default=8)
    parser.add_argument('--eval-per-city', type=int, default=12)
    args = parser.parse_args()
    if not (1 <= args.steps <= 120 and 1 <= args.train_per_city <= 12 and 2 <= args.eval_per_city <= 16):
        parser.error('caps: 120 steps, 12 train paths/city, 16 eval paths/city')
    args.seed, args.map_checkpoint = 17, None
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.20)
    torch.set_float32_matmul_precision('high')
    backbone = load_encoder(BACKBONE).to(DEVICE).eval().requires_grad_(False)
    data = collect(args, backbone)
    rng = np.random.default_rng(args.seed)
    draws = {split: rng.integers(len(clips), size=(2000, len(clips))) for split, clips in data.items()}
    results: dict[str, object] = {'config': vars(args)}
    variants = {}
    results['variants'] = variants
    for seed in (17, 29):
        for name, mode, supervised, gate in VARIANTS:
            model, slot_head = fit(backbone, data['train'], args.steps, seed, mode, supervised, gate)
            scores = {split: measure(model, slot_head, clips, draws[split]) for split, clips in data.items()}
            variants.setdefault(name, {})[str(seed)] = scores
            for head, rows in scores['test'].items():
                emit(f'{seed} {name} {head}: ' + ', '.join(f'{gap}={row["accuracy"]:.1%}' for gap, row in rows.items()))
            results['peak_torch_cuda_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            (args.out / 'ablation.json').write_text(json.dumps(results, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
