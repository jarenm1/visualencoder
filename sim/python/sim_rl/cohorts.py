"""Paired near/hidden/far evaluations under the same sustained-hold objective."""
from typing import Any

import torch

from .cli import (TEST_SEED, TEST_STYLE, build_policy, check_episodes, emit, load_checkpoint,
                  sha256, validate_checkpoint)
from .curriculum import UNIFORM_WEIGHTS
from .evaluation import evaluate
from .features import FrozenFeatures, precision_of
from .native import TASK, CudaEnv


def run(args):
    checkpoints = [(str(path), load_checkpoint(path), False) for path in args.checkpoint]
    checkpoints += [(str(path), load_checkpoint(path, for_task_transfer=True), True)
                    for path in args.baseline_checkpoint]
    first = checkpoints[0][1]['metadata']
    features = FrozenFeatures(first['encoder'], torch.device('cuda:0'), precision_of(first['preprocessing']))
    results: dict[str, dict[str, Any]] = {name: {'sha256': sha256(name), 'selected_steps': saved['metadata']['steps'],
                      'source_task': saved['metadata']['task'], 'evaluation_task': TASK,
                      'explicit_task_transfer': transfer, 'cohorts': {}}
               for name, saved, transfer in checkpoints}
    # Keep the standard hidden range separate from the genuinely out-of-training-range cohort.
    for distance, cohorts in ((None, [('standard', UNIFORM_WEIGHTS), ('hidden', (0., 0., 1.))]),
                              ((35., 50.), [('far', UNIFORM_WEIGHTS), ('hidden_far', (0., 0., 1.))])):
        with CudaEnv(args.library, args.eval_envs, TEST_SEED, first['width'], first['height'], TEST_STYLE,
                     first['control_mode'], args.city, start_distance=distance) as env:
            with env.execution():
                for name, saved, transfer in checkpoints:
                    validate_checkpoint(saved, env, features, for_task_transfer=transfer)
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
