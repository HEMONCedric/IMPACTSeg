#!/usr/bin/env python3
"""Portable launcher: validate splits, resolve a template, then invoke KonfAI."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs'/'train_2p5d.yml')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--splits-root', type=Path, default=ROOT/'splits')
    parser.add_argument('--split-name', type=str, default='fold1')
    parser.add_argument('--dataset-format', type=str, default=None)
    parser.add_argument('--modalities', nargs='+', default=None)
    parser.add_argument('--train-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max-iterations', type=int, default=None)
    parser.add_argument('--memory-budget', type=str, default=None)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--resume-checkpoint',
        type=Path,
        default=None,
        help='Resume the complete KonfAI state from this checkpoint (model, optimizer, LR and scheduler).',
    )
    parser.add_argument('--skip-split-check', action='store_true')
    parser.add_argument('--init-only', action='store_true')
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    splits_root = args.splits_root.resolve()
    run_root = args.run_root.resolve()
    config = args.config.resolve()
    resume_checkpoint = args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {resume_checkpoint}')
    split_dir = splits_root/args.split_name
    if not args.skip_split_check:
        subprocess.run([sys.executable, str(ROOT/'tools'/'validate_splits.py'), '--split-dir', str(split_dir)], check=True)
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = run_root/'resolved_config.yml'
    resolve_command = [
        sys.executable, str(ROOT/'tools'/'resolve_config.py'), '--template', str(config), '--output', str(resolved),
        '--dataset-root', str(dataset_root), '--splits-root', str(splits_root), '--split-name', args.split_name,
    ]
    if args.dataset_format is not None:
        resolve_command.extend(['--dataset-format', args.dataset_format])
    if args.modalities is not None:
        resolve_command.extend(['--modalities', *args.modalities])
    if args.train_name is not None:
        resolve_command.extend(['--train-name', args.train_name])
    if args.epochs is not None:
        resolve_command.extend(['--epochs', str(args.epochs)])
    if args.memory_budget is not None:
        resolve_command.extend(['--memory-budget', args.memory_budget])
    subprocess.run(resolve_command, check=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if args.max_iterations is not None:
        if args.max_iterations <= 0:
            raise SystemExit('--max-iterations must be strictly positive.')
        env['IMPACTSEG_MAX_ITERATIONS'] = str(args.max_iterations)
    (run_root/'launch_settings.json').write_text(json.dumps({
        'config_template': str(config),
        'resolved_config': str(resolved),
        'dataset_root': str(dataset_root),
        'dataset_format': args.dataset_format,
        'modalities': args.modalities,
        'split_dir': str(split_dir),
        'train_name': args.train_name,
        'epochs_floor': args.epochs,
        'max_iterations': args.max_iterations,
        'memory_budget': args.memory_budget,
        'gpu': args.gpu,
        'resume_checkpoint': str(resume_checkpoint) if resume_checkpoint is not None else None,
        'init_only': args.init_only,
    }, indent=2) + '\n')
    workflow = 'RESUME' if resume_checkpoint is not None else 'TRAIN'
    command = [sys.executable, '-m', 'impactseg.konfai_train', workflow, '-y', '--gpu', str(args.gpu), '--config', str(resolved),
               '--checkpoints-dir', str(run_root/'checkpoints'), '--statistics-dir', str(run_root/'statistics')]
    if resume_checkpoint is not None:
        # Intentionally omit --lr: KonfAI restores the optimizer LR and
        # continues the scheduler from its saved update counter.
        command.extend(['--model', str(resume_checkpoint)])
    if args.init_only:
        command.append('--init')
    subprocess.run(command, check=True, cwd=run_root, env=env)


if __name__ == '__main__':
    main()
