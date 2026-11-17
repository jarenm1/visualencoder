"""Paired hidden/far evaluations; identical episode catalogs and no intrinsic rewards."""
from typing import Any

import torch

from .cli import (TEST_SEED, TEST_STYLE, build_policy, check_episodes, emit, load_checkpoint,
                  sha256, validate_checkpoint)
from .curriculum import UNIFORM_WEIGHTS
from .evaluation import evaluate
from .features import FrozenFeatures, precision_of
from .native import CudaEnv


def run(args):
    checkpoints = [(str(path), load_checkpoint(path)) for path in args.checkpoint]
    first = checkpoints[0][1]['metadata']
    features = FrozenFeatures(first['encoder'], torch.device('cuda:0'), precision_of(first['preprocessing']))
    results: dict[str, dict[str, Any]] = {name: {'sha256': sha256(name), 'selected_steps': saved['metadata']['steps'],
                      'reward_augmentation': saved['metadata'].get('reward_augmentation'), 'cohorts': {}}
               for name, saved in checkpoints}
    # Keep the standard hidden range separate from the genuinely out-of-training-range cohort.
    for distance, cohorts in ((None, [('hidden', (0., 0., 1.))]),
                              ((35., 50.), [('far', UNIFORM_WEIGHTS), ('hidden_far', (0., 0., 1.))])):
        with CudaEnv(args.library, args.eval_envs, TEST_SEED, first['width'], first['height'], TEST_STYLE,
                     first['control_mode'], args.city, start_distance=distance) as env:
            with env.execution():
                for name, saved in checkpoints:
                    validate_checkpoint(saved, env, features)
                    policy = build_policy(env, features)
                    policy.load_state_dict(saved['model'])
                    for cohort, weights in cohorts:
                        cell = {}
                        for mode, no_gps in (('gps', 0.), ('no_gps', 1.)):
                            cell[mode] = check_episodes(evaluate(
                                env, features, policy=policy, seed=TEST_SEED, no_gps=no_gps,
                                goal_refresh_steps=args.goal_refresh_steps, bucket_weights=weights))
                        results[name]['cohorts'][cohort] = cell
                        emit({'seed': TEST_SEED, 'goal_refresh_steps': args.goal_refresh_steps,
                              'results': results}, args.out)
                        print({'checkpoint': name, 'cohort': cohort,
                               'success': {mode: value['success'] for mode, value in cell.items()}}, flush=True)
                    del policy
    return results
