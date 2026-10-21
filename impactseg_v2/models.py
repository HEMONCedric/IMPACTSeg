"""KonfAI 1.8 model preserving the original ResEncMCT topology.

KonfAI 1.8 already ships a weight-exact nnU-Net ResidualEncoderUNet.  This
variant exposes the named logits/probability supervision heads used by the
IMPACTSeg criteria, preserves historical BatchNorm for the 2.5D experiments,
and supports configurable normalization for 3D experiments. The topology,
strides and convolution order remain the nnU-Net ResEnc implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.config import config

from .labels import FINE_NETWORK_OUTPUTS, NOT_PREDICTED_FINE_IDS


def _validate_topology(name: str, dim: int, n_stages: int, features: list[int], strides: list[int | list[int]]) -> None:
    """Local validation: avoid importing KonfAI's private nnU-Net helper module."""
    if dim not in (2, 3):
        raise ValueError(f"{name} supports dim 2 or 3, got {dim}.")
    if n_stages < 2 or len(features) != n_stages or len(strides) != n_stages:
        raise ValueError("Topology needs >=2 stages and one feature width/stride per stage.")


def _as_stage_list(value: int | list[int], length: int, name: str) -> list[int]:
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length or any(item < 1 for item in values):
        raise ValueError(f"{name} must have {length} positive entries, got {values}.")
    return values


def _as_kernel_list(kernel_sizes: int | list[Any], n_stages: int) -> list[Any]:
    if isinstance(kernel_sizes, int):
        return [kernel_sizes] * n_stages
    if len(kernel_sizes) != n_stages:
        raise ValueError(f"kernel_sizes must have {n_stages} entries, got {kernel_sizes}.")
    return list(kernel_sizes)


def _padding(kernel_size: int | list[int]) -> int | list[int]:
    if isinstance(kernel_size, int):
        return (kernel_size - 1) // 2
    return [(int(value) - 1) // 2 for value in kernel_size]


def _decoder_block_config(
    stride: int | list[int], kernel_size: int | list[int], conv_bias: bool, negative_slope: float, norm_mode: str
) -> blocks.BlockConfig:
    return blocks.BlockConfig(
        kernel_size=kernel_size,
        stride=stride,
        padding=_padding(kernel_size),
        bias=conv_bias,
        activation=f"LeakyReLU;{negative_slope}",
        norm_mode=norm_mode,
    )


class _ConfidenceArgmax(torch.nn.Module):
    """Argmax that rejects low-confidence foreground voxels as background."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        confidence, labels = torch.max(probabilities, dim=1, keepdim=True)
        return labels.masked_fill((labels != 0) & (confidence < self.threshold), 0)


class _ArgmaxConfidence(torch.nn.Module):
    """Pack argmax labels and 8-bit confidence into two channels."""

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape[1] == FINE_NETWORK_OUTPUTS:  # fine map: classes not provided by the released models
            probabilities = probabilities.clone()
            probabilities[:, list(NOT_PREDICTED_FINE_IDS)] = 0
        confidence, labels = torch.max(probabilities, dim=1, keepdim=True)
        labels = labels.to(torch.uint8)
        confidence_u8 = torch.round(confidence * 255.0).to(torch.uint8)
        return torch.cat((labels, confidence_u8), dim=1)


@config()
class ImpactSegResidualEncoderUNet(network.Network):
    """2D/2.5D or 3D residual U-Net with named logits/probability DS heads."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|PolynomialLR": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {
            "default": network.TargetCriterionsLoader()
        },
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        dim: int = 2,
        in_channels: int = 5,
        n_stages: int = 6,
        features_per_stage: list[int] = [32, 64, 128, 256, 320, 512],
        strides: list[int | list[int]] = [1, 2, 2, 2, 2, 2],
        kernel_sizes: int | list[Any] = 3,
        n_blocks_per_stage: int | list[int] = [1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder: int | list[int] = [1, 1, 1, 1, 1],
        num_classes: int = 132,
        conv_bias: bool = True,
        negative_slope: float = 1e-2,
        norm_mode: str = "BATCH",
        deep_supervision_outputs: int = 3,
        confidence_threshold: float | None = None,
        emit_confidence: bool = False,
    ) -> None:
        _validate_topology("ImpactSegResidualEncoderUNet", dim, n_stages, features_per_stage, strides)
        if norm_mode not in blocks.NormMode.__members__:
            raise ValueError(f"Unknown KonfAI norm_mode={norm_mode!r}")
        if not 1 <= deep_supervision_outputs <= n_stages - 1:
            raise ValueError("deep_supervision_outputs must be in [1, n_stages - 1]")
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if confidence_threshold is not None and emit_confidence:
            raise ValueError("confidence_threshold and emit_confidence are mutually exclusive")

        kernel_list = _as_kernel_list(kernel_sizes, n_stages)
        encoder_blocks = _as_stage_list(n_blocks_per_stage, n_stages, "n_blocks_per_stage")
        decoder_blocks = _as_stage_list(n_conv_per_stage_decoder, n_stages - 1, "n_conv_per_stage_decoder")
        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
        )

        self.add_module(
            "Stem",
            blocks.ConvBlock(
                in_channels=in_channels,
                out_channels=features_per_stage[0],
                block_configs=[
                    _decoder_block_config(1, kernel_list[0], conv_bias, negative_slope, norm_mode)
                ],
                dim=dim,
            ),
            in_branch=[0],
            out_branch=["stem"],
        )
        for stage in range(n_stages):
            stage_input = features_per_stage[0] if stage == 0 else features_per_stage[stage - 1]
            for block in range(encoder_blocks[stage]):
                source = "stem" if stage == 0 and block == 0 else f"enc{stage - 1}" if block == 0 else f"enc{stage}"
                self.add_module(
                    f"Encoder_{stage}_Block_{block}",
                    blocks.ResidualBlockD(
                        in_channels=stage_input if block == 0 else features_per_stage[stage],
                        out_channels=features_per_stage[stage],
                        dim=dim,
                        kernel_size=kernel_list[stage],
                        stride=strides[stage] if block == 0 else 1,
                        conv_bias=conv_bias,
                        negative_slope=negative_slope,
                        norm_mode=norm_mode,
                    ),
                    in_branch=[source],
                    out_branch=[f"enc{stage}"],
                )

        # Coarsest-to-finest decoder.  The last N heads are the historical
        # final + auxiliary deep-supervision outputs, exposed both as logits
        # and softmax probabilities for the CE and Dice criteria respectively.
        for decoder_stage in range(n_stages - 1):
            below = n_stages - 1 - decoder_stage
            skip = n_stages - 2 - decoder_stage
            upsample_stride: Any = strides[below]
            self.add_module(
                f"Up_{decoder_stage}",
                blocks.get_torch_module("ConvTranspose", dim)(
                    in_channels=features_per_stage[below],
                    out_channels=features_per_stage[skip],
                    kernel_size=upsample_stride,
                    stride=upsample_stride,
                    padding=0,
                    bias=conv_bias,
                ),
                in_branch=[f"enc{n_stages - 1}" if decoder_stage == 0 else f"dec{decoder_stage - 1}"],
                out_branch=[f"up{decoder_stage}"],
            )
            self.add_module(
                f"Skip_{decoder_stage}",
                blocks.Concat(),
                in_branch=[f"up{decoder_stage}", f"enc{skip}"],
                out_branch=[f"up{decoder_stage}"],
            )
            self.add_module(
                f"Decoder_{decoder_stage}",
                blocks.ConvBlock(
                    in_channels=2 * features_per_stage[skip],
                    out_channels=features_per_stage[skip],
                    block_configs=[
                        _decoder_block_config(1, kernel_list[skip], conv_bias, negative_slope, norm_mode)
                        for _ in range(decoder_blocks[decoder_stage])
                    ],
                    dim=dim,
                ),
                in_branch=[f"up{decoder_stage}"],
                out_branch=[f"dec{decoder_stage}"],
            )
            if decoder_stage >= (n_stages - 1 - deep_supervision_outputs):
                self.add_module(
                    f"Logits_{decoder_stage}",
                    blocks.get_torch_module("Conv", dim)(
                        features_per_stage[skip], num_classes, kernel_size=1, stride=1, padding=0, bias=True
                    ),
                    in_branch=[f"dec{decoder_stage}"],
                    out_branch=[f"logits{decoder_stage}", -1],
                )
                self.add_module(
                    f"Softmax_{decoder_stage}",
                    torch.nn.Softmax(dim=1),
                    in_branch=[f"logits{decoder_stage}"],
                    out_branch=[f"probabilities{decoder_stage}", -1],
                )

        final_stage = n_stages - 2
        self.add_module(
            "Argmax",
            blocks.ArgMax(dim=1),
            in_branch=[f"logits{final_stage}"],
            out_branch=[-1],
        )
        if confidence_threshold is not None:
            self.add_module(
                "ConfidenceArgmax",
                _ConfidenceArgmax(confidence_threshold),
                in_branch=[f"probabilities{final_stage}"],
                out_branch=[-1],
            )
        elif emit_confidence:
            self.add_module(
                "ArgmaxConfidence",
                _ArgmaxConfidence(),
                in_branch=[f"probabilities{final_stage}"],
                out_branch=[-1],
            )
