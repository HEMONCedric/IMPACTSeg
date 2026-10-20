"""Inference-only extensions for IMPACTSeg.

KonfAI's legacy ``overlap: None`` grid keeps every border patch full-sized by
spreading the remainder over the grid.  In KonfAI 1.8, however, ``None`` also
disables the output combine stage, so overlapping label patches are assembled
with last-write-wins semantics. ``ConfidenceSelectOutputDataset`` instead
retains, at each overlapping voxel, the label predicted with the highest
softmax confidence. Historical trim experiments are archived separately.
"""

from __future__ import annotations

from typing import Any

import torch

from konfai.data.patching import Accumulator
from konfai.predictor import OutputDataset


def __getattr__(name: str) -> Any:
    """Replay old run-local YAMLs without keeping trim in the production code."""
    if name == "AdaptiveAutoTrimOutputDataset":
        import warnings
        from legacy.experimental.inference import AdaptiveAutoTrimOutputDataset
        warnings.warn(
            "AdaptiveAutoTrimOutputDataset is archived; use "
            "legacy.experimental.inference:AdaptiveAutoTrimOutputDataset for historical replays.",
            DeprecationWarning, stacklevel=2,
        )
        return AdaptiveAutoTrimOutputDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _ConfidenceSelectAccumulator(Accumulator):
    """Keep, in overlap regions, the label from the most confident patch."""

    def _blend(
        self,
        layer: torch.Tensor,
        patch_slice: tuple[slice, ...],
        row_offset: int = 0,
    ) -> None:
        n = self._n
        data = layer
        for dim, item in enumerate(patch_slice):
            if item.stop - item.start == 1:
                data = data.unsqueeze(dim=dim + n)

        dest = [slice(item.start, min(item.stop, self.shape[dim])) for dim, item in enumerate(patch_slice)]
        data = data[tuple([slice(None)] * n + [slice(0, item.stop - item.start) for item in dest])]
        slices_dest = tuple([slice(None)] * n + dest)
        if self._result is None:
            self._result = torch.zeros(
                [2, *self.shape],
                dtype=data.dtype,
                device=data.device,
            )
        result = self._result
        current = result[slices_dest]
        labels = data[0:1]
        scores = data[1:2]
        current_labels = current[0:1]
        current_scores = current[1:2]
        torch.where(scores >= current_scores, labels, current_labels, out=current_labels)
        torch.maximum(current_scores, scores, out=current_scores)

    def assemble(self) -> torch.Tensor:
        if self._result is None:
            return super().assemble()
        labels = self._result[0:1]
        self._reset()
        return labels


class _ProbabilityConfidenceSelectAccumulator(_ConfidenceSelectAccumulator):
    """Convert an ensemble mean-probability patch before overlap selection."""

    def _blend(
        self,
        layer: torch.Tensor,
        patch_slice: tuple[slice, ...],
        row_offset: int = 0,
    ) -> None:
        confidence, labels = torch.max(layer, dim=0, keepdim=True)
        packed = torch.cat(
            (
                labels.to(torch.uint8),
                torch.round(confidence * 255.0).to(torch.uint8),
            ),
            dim=0,
        )
        super()._blend(packed, patch_slice, row_offset=row_offset)


class ConfidenceSelectOutputDataset(OutputDataset):
    """Assemble packed label/confidence patches by maximum confidence."""

    accumulator_type: type[Accumulator] = _ConfidenceSelectAccumulator

    def set_patch_config(
        self,
        patch_size: list[int] | None,
        overlap: int | float | str | list[int | float | str] | None,
        nb_data_augmentation: int,
    ) -> None:
        super().set_patch_config(patch_size, overlap, nb_data_augmentation)
        self._streaming_enabled = False

    def _ensure_case_state(
        self,
        index_dataset: int,
        index_augmentation: int,
        layer: torch.Tensor,
        dataset: Any,
        attribute: Any,
        number_of_channels_per_model: list[int] | None,
    ) -> None:
        existed = (
            index_dataset in self.output_layer_accumulator
            and index_augmentation in self.output_layer_accumulator[index_dataset]
        )
        super()._ensure_case_state(
            index_dataset,
            index_augmentation,
            layer,
            dataset,
            attribute,
            number_of_channels_per_model,
        )
        if existed:
            return
        input_dataset = dataset.get_dataset_from_index(self.group_dest, index_dataset)
        patch_slices = input_dataset.patch.get_patch_slices(index_augmentation)
        self.output_layer_accumulator[index_dataset][index_augmentation] = self.accumulator_type(
            patch_slices,
            input_dataset.patch.patch_size,
            None,
            batch=False,
            sweep_axis=input_dataset.patch.get_sweep_axis(index_augmentation),
        )


class ProbabilityConfidenceSelectOutputDataset(ConfidenceSelectOutputDataset):
    """Pack mean probabilities on-device, then resolve overlaps by confidence.

    ``ModelComposite`` has already averaged the models when ``add_layer`` is
    called. Packing here, while the patch is still on the GPU, avoids moving
    every class-probability channel to the host merely to compute an argmax.
    """

    accumulator_type = _ConfidenceSelectAccumulator

    def add_layer(
        self,
        index_dataset: int,
        index_augmentation: int,
        index_patch: int,
        layer: torch.Tensor,
        dataset: Any,
        attribute: Any = None,
        number_of_channels_per_model: list[int] | None = None,
    ) -> Any:
        confidence, labels = torch.max(layer, dim=0, keepdim=True)
        packed = torch.cat(
            (
                labels.to(torch.uint8),
                torch.round(confidence * 255.0).to(torch.uint8),
            ),
            dim=0,
        )
        return super().add_layer(
            index_dataset,
            index_augmentation,
            index_patch,
            packed,
            dataset,
            attribute,
            number_of_channels_per_model,
        )
