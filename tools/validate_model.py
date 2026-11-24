#!/usr/bin/env python3
"""Instantiate the ResEnc model and run forward, loss, backward and SGD safely."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from impactseg.labels import FINE_NETWORK_OUTPUTS  # noqa: E402
from impactseg.losses import DiceNoBackground, DiceNoBackgroundAbsentAware, SegCrossEntropy  # noqa: E402
from impactseg.models import ImpactSegResidualEncoderUNet  # noqa: E402
from konfai.network import network  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--spatial-size', type=int, default=64)
    parser.add_argument('--dim', type=int, choices=(2, 3), default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--norm-mode', default='BATCH')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this GPU validation.')
    if args.spatial_size % 32:
        raise SystemExit('--spatial-size must be divisible by the 5 downsampling stages (32).')
    if args.batch_size <= 0:
        raise SystemExit('--batch-size must be positive.')
    torch.manual_seed(42)
    model = ImpactSegResidualEncoderUNet(
        dim=args.dim, in_channels=5 if args.dim == 2 else 1,
        norm_mode=args.norm_mode, num_classes=FINE_NETWORK_OUTPUTS,
    )
    network.Network.to(model, args.gpu)
    model.train()
    device = torch.device(f'cuda:{args.gpu}')
    spatial = (args.spatial_size,) * args.dim
    image = torch.randn(args.batch_size, 5 if args.dim == 2 else 1, *spatial, device=device)
    target = torch.randint(0, FINE_NETWORK_OUTPUTS, (args.batch_size, 1, *spatial), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = dict(model.named_forward(image))
        loss = (
            SegCrossEntropy(background_weight=0.2, label_smoothing=0.02)(outputs['Logits_4'], target)
            + DiceNoBackground(only_present=True)(outputs['Softmax_4'], target)
            + DiceNoBackgroundAbsentAware(coefficient=0.20)(outputs['Softmax_4'], target)
            + 0.20 * DiceNoBackground(only_present=True)(outputs['Softmax_3'], target)
            + 0.10 * DiceNoBackground(only_present=True)(outputs['Softmax_2'], target)
        )
    if not torch.isfinite(loss):
        raise SystemExit(f'Non-finite loss: {loss.detach().item()}')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    if not gradients_finite:
        raise SystemExit('Non-finite gradients after AMP unscaling.')
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    report = {
        'status': 'PASS', 'device': torch.cuda.get_device_name(args.gpu), 'network_outputs': FINE_NETWORK_OUTPUTS,
        'parameters': sum(parameter.numel() for parameter in model.parameters()), 'loss': float(loss.detach().cpu()),
        'dim': args.dim, 'batch_size': args.batch_size, 'norm_mode': args.norm_mode,
        'gradients_finite': gradients_finite,
        'step_seconds': time.perf_counter() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3,
        'outputs': {name: list(outputs[name].shape) for name in ('Logits_2', 'Logits_3', 'Logits_4', 'Softmax_2', 'Softmax_3', 'Softmax_4', 'Argmax')},
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
