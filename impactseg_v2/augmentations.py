"""IMPACTSeg-specific augmentation distributions on the public KonfAI 1.8 API."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from konfai.data.augmentation import DataAugmentation
from konfai.data.augmentation.base import _hashed_normal_field, _rotation_2d_matrix, _rotation_3d_matrix
from konfai.data.augmentation.placed import PlacedDraw
from konfai.data.augmentation.spatial import EulerTransform
from konfai.data.transform import LocalityKind, PatchLocality, RegionContext
from konfai.utils.dataset import Attribute


def _kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()


class SafeRotate(EulerTransform):
    """Old radians-based rotation; 2.5D data rotate in Y/X only.

    KonfAI provides probability, lifecycle and streamed REGRID machinery. Only
    this sampling restriction remains custom: native Rotate samples all axes
    of a 3D source, whereas the historical experiment had in_plane_only.
    """

    def __init__(
        self,
        angle_min: float = -0.174533,
        angle_max: float = 0.174533,
        in_plane_only: bool = False,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__()
        del groups  # Geometry must be identically applied to input and target.
        if angle_max < angle_min:
            raise ValueError("angle_max must be >= angle_min")
        self.angle_min, self.angle_max = float(angle_min), float(angle_max)
        self.in_plane_only = bool(in_plane_only)

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        dim = len(shapes[0])
        if dim not in (2, 3):
            raise ValueError(f"SafeRotate supports 2D/3D spatial shapes, got {shapes[0]}")
        angles = torch.rand((len(shapes), 1 if dim == 2 or self.in_plane_only else dim))
        angles = angles * (self.angle_max - self.angle_min) + self.angle_min
        if dim == 2:
            matrices = [_rotation_2d_matrix(angle).unsqueeze(0) for angle in angles]
        elif self.in_plane_only:
            matrices = [
                _rotation_3d_matrix(torch.tensor([0.0, 0.0, angle[0]], dtype=torch.float32)).unsqueeze(0)
                for angle in angles
            ]
        else:
            matrices = [_rotation_3d_matrix(angle).unsqueeze(0) for angle in angles]
        self.matrix[index] = matrices
        return shapes


class GaussianNoise(PlacedDraw):
    """Historical Gaussian noise, made deterministic per case/copy for streaming."""

    def __init__(self, std_min: float = 0.0, std_max: float = 0.1, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        if std_max < std_min:
            raise ValueError("std_max must be >= std_min")
        self.std_min, self.std_max = float(std_min), float(std_max)
        self.stds: dict[int, list[float]] = {}
        self.seeds: dict[int, list[int]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.stds[index] = (self.std_min + torch.rand(len(shapes)) * (self.std_max - self.std_min)).tolist()
        self.seeds[index] = [int(torch.randint(0, 2**31 - 1, (1,))) for _ in shapes]
        return shapes

    def _apply(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        std = self.stds[index][a]
        if std == 0:
            return tensor
        field = _hashed_normal_field(self.seeds[index][a], tuple(tensor.shape), offsets, full, tensor.device)
        return (tensor.float() + field * std).to(tensor.dtype)


class GaussianBlur(DataAugmentation):
    """Depthwise Gaussian blur with a streaming-safe dynamic halo.

    In 2.5D mode each axial slice is blurred independently. The input keeps
    exactly its five requested slices and only the in-plane axes receive a
    halo.
    """

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 1.0,
        in_plane_only: bool = False,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if sigma_max < sigma_min:
            raise ValueError("sigma_max must be >= sigma_min")
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.in_plane_only = bool(in_plane_only)
        self.sigmas: dict[int, list[float]] = {}
        self.spatial_dims: dict[int, list[int]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.sigmas[index] = (self.sigma_min + torch.rand(len(shapes)) * (self.sigma_max - self.sigma_min)).tolist()
        self.spatial_dims[index] = [len(shape) for shape in shapes]
        return shapes

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        del cache_attribute
        radius = max(1, int(math.ceil(3 * self.sigmas[index][a])))
        dims = self.spatial_dims[index][a]
        halo = (0, *([radius] * (dims - 1))) if self.in_plane_only and dims >= 3 else (radius,) * dims
        return PatchLocality(LocalityKind.HALO, halo=halo)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        sigma = self.sigmas[index][a]
        if sigma == 0:
            return tensor
        dim = tensor.ndim - 1
        line = _kernel_1d(sigma, tensor.device)
        radius = line.numel() // 2

        if dim == 2 or (dim == 3 and self.in_plane_only):
            kernel = torch.einsum("i,j->ij", line, line)
            kernel = kernel.reshape(1, 1, *kernel.shape).repeat(tensor.shape[0], 1, 1, 1)
            if dim == 2:
                batch = tensor.unsqueeze(0)
                result = functional.conv2d(
                    batch.float(), kernel, padding=radius, groups=tensor.shape[0]
                )
                return result.squeeze(0).to(tensor.dtype)

            slices = tensor.permute(1, 0, 2, 3)
            result = functional.conv2d(
                slices.float(), kernel, padding=radius, groups=tensor.shape[0]
            )
            return result.permute(1, 0, 2, 3).to(tensor.dtype)

        if dim == 3:
            kernel = torch.einsum("i,j,k->ijk", line, line, line)
            kernel = kernel.reshape(1, 1, *kernel.shape).repeat(tensor.shape[0], 1, 1, 1, 1)
            return functional.conv3d(
                tensor.unsqueeze(0).float(), kernel, padding=radius, groups=tensor.shape[0]
            ).squeeze(0).to(tensor.dtype)

        raise ValueError(f"GaussianBlur expects 2D/3D tensors, got {tuple(tensor.shape)}")

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor


class SimulateLowResolution(DataAugmentation):
    """Simulate a lower in-plane acquisition resolution on image tensors.

    KonfAI 1.8 ships spatial scaling but no image-only downsample/upsample
    augmentation. This draw keeps the label grid untouched and, for 2.5D
    volumes, resamples each axial slice independently: neighbouring slices are
    never mixed before ``SelectRelativeSliceOffsets`` assembles the channels.

    This is a REGRID rather than a pointwise draw: every output voxel depends
    on interpolation taps on a lower-resolution grid. The streaming methods
    anchor both interpolation grids to the complete case and pull only the
    source box needed by the requested patch.
    """

    locality = LocalityKind.REGRID

    def __init__(
        self,
        factor_min: float = 1.0,
        factor_max: float = 2.0,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if factor_min < 1.0:
            raise ValueError("factor_min must be >= 1.0 (only degrade resolution)")
        if factor_max < factor_min:
            raise ValueError("factor_max must be >= factor_min")
        self.factor_min, self.factor_max = float(factor_min), float(factor_max)
        self.factors: dict[int, list[float]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.factors[index] = (self.factor_min + torch.rand(len(shapes)) * (self.factor_max - self.factor_min)).tolist()
        return shapes

    @staticmethod
    def _low_extent(extent: int, factor: float) -> int:
        return max(1, round(extent / factor))

    @classmethod
    def _axis_pull(cls, target: slice, extent: int, factor: float) -> tuple[slice, slice]:
        """Return the low-grid and source intervals required by a target interval."""
        if target.start is None or target.stop is None or target.stop <= target.start:
            raise ValueError(f"Expected a non-empty bounded target slice, got {target}")
        low_extent = cls._low_extent(extent, factor)
        if low_extent == extent:
            same = slice(target.start, target.stop)
            return same, same

        # align_corners=False maps o to (o + .5) * input/output - .5.
        low_first = (target.start + 0.5) * low_extent / extent - 0.5
        low_last = (target.stop - 0.5) * low_extent / extent - 0.5
        low_start = max(0, min(low_extent - 1, math.floor(low_first)))
        low_stop = max(low_start + 1, min(low_extent, math.floor(low_last) + 2))

        source_first = (low_start + 0.5) * extent / low_extent - 0.5
        source_last = (low_stop - 0.5) * extent / low_extent - 0.5
        source_start = max(0, min(extent - 1, math.floor(source_first)))
        source_stop = max(source_start + 1, min(extent, math.floor(source_last) + 2))
        return slice(low_start, low_stop), slice(source_start, source_stop)

    def _stream_region_source(
        self,
        index: int,
        a: int,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
    ) -> list[slice]:
        if len(source_spatial_shape) not in (2, 3):
            raise ValueError(f"SimulateLowResolution expects a 2D/3D grid, got {source_spatial_shape}")
        factor = self.factors[index][a]
        source = list(target_slices)
        for axis in (-2, -1):
            _, source[axis] = self._axis_pull(target_slices[axis], source_spatial_shape[axis], factor)
        return source

    @staticmethod
    def _normalised_grid_axis(
        coordinates: torch.Tensor, block_start: int, block_extent: int
    ) -> torch.Tensor:
        return 2.0 * (coordinates - float(block_start) + 0.5) / float(block_extent) - 1.0

    def _stream_region(
        self,
        name: str,
        index: int,
        a: int,
        tensor: torch.Tensor,
        context: RegionContext,
    ) -> torch.Tensor:
        del name
        factor = self.factors[index][a]
        full_height, full_width = (int(value) for value in context.source_shape[-2:])
        low_height = self._low_extent(full_height, factor)
        low_width = self._low_extent(full_width, factor)
        if low_height == full_height and low_width == full_width:
            return tensor

        low_y, _ = self._axis_pull(context.target[-2], full_height, factor)
        low_x, _ = self._axis_pull(context.target[-1], full_width, factor)
        source_y, source_x = context.source[-2:]

        if tensor.ndim == 3:  # [C, H, W]
            batch = tensor.unsqueeze(0)
            is_volume = False
        elif tensor.ndim == 4:  # [C, Z, H, W], process each Z independently.
            batch = tensor.permute(1, 0, 2, 3)
            is_volume = True
        else:
            raise ValueError(
                f"SimulateLowResolution expects [C,H,W] or [C,Z,H,W], got {tuple(tensor.shape)}"
            )

        device = tensor.device
        low_y_indices = torch.arange(low_y.start, low_y.stop, device=device, dtype=torch.float32)
        low_x_indices = torch.arange(low_x.start, low_x.stop, device=device, dtype=torch.float32)
        source_y_coordinates = (low_y_indices + 0.5) * full_height / low_height - 0.5
        source_x_coordinates = (low_x_indices + 0.5) * full_width / low_width - 0.5
        down_y = self._normalised_grid_axis(source_y_coordinates, source_y.start, batch.shape[-2])
        down_x = self._normalised_grid_axis(source_x_coordinates, source_x.start, batch.shape[-1])
        down_grid_y, down_grid_x = torch.meshgrid(down_y, down_x, indexing="ij")
        down_grid = (
            torch.stack((down_grid_x, down_grid_y), dim=-1)
            .unsqueeze(0)
            .expand(batch.shape[0], -1, -1, -1)
        )
        low = functional.grid_sample(
            batch.float(), down_grid, mode="bilinear", padding_mode="border", align_corners=False
        )

        target_y_indices = torch.arange(
            context.target[-2].start, context.target[-2].stop, device=device, dtype=torch.float32
        )
        target_x_indices = torch.arange(
            context.target[-1].start, context.target[-1].stop, device=device, dtype=torch.float32
        )
        low_y_coordinates = (target_y_indices + 0.5) * low_height / full_height - 0.5
        low_x_coordinates = (target_x_indices + 0.5) * low_width / full_width - 0.5
        up_y = self._normalised_grid_axis(low_y_coordinates, low_y.start, low.shape[-2])
        up_x = self._normalised_grid_axis(low_x_coordinates, low_x.start, low.shape[-1])
        up_grid_y, up_grid_x = torch.meshgrid(up_y, up_x, indexing="ij")
        up_grid = (
            torch.stack((up_grid_x, up_grid_y), dim=-1)
            .unsqueeze(0)
            .expand(low.shape[0], -1, -1, -1)
        )
        restored = functional.grid_sample(
            low, up_grid, mode="bilinear", padding_mode="border", align_corners=False
        ).to(tensor.dtype)
        return restored.permute(1, 0, 2, 3) if is_volume else restored.squeeze(0)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        factor = self.factors[index][a]
        if factor == 1.0:
            return tensor
        if tensor.ndim == 3:  # [C, H, W]
            height, width = tensor.shape[-2:]
            low_shape = (max(1, round(height / factor)), max(1, round(width / factor)))
            low = functional.interpolate(tensor.unsqueeze(0).float(), size=low_shape, mode="bilinear", align_corners=False)
            return functional.interpolate(low, size=(height, width), mode="bilinear", align_corners=False).squeeze(0).to(tensor.dtype)
        if tensor.ndim == 4:  # [C, Z, H, W]: degrade each Z slice independently.
            _, _, height, width = tensor.shape
            low_shape = (max(1, round(height / factor)), max(1, round(width / factor)))
            slices = tensor.float().permute(1, 0, 2, 3)
            low = functional.interpolate(slices, size=low_shape, mode="bilinear", align_corners=False)
            restored = functional.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
            return restored.permute(1, 0, 2, 3).to(tensor.dtype)
        raise ValueError(f"SimulateLowResolution expects [C,H,W] or [C,Z,H,W], got {tuple(tensor.shape)}")

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor


class Gamma(DataAugmentation):
    """Historical min/max-normalised gamma draw."""

    def __init__(
        self, gamma_min: float = 0.7, gamma_max: float = 1.5, groups: list[str] | None = None, eps: float = 1e-6
    ) -> None:
        super().__init__(groups)
        if gamma_max < gamma_min:
            raise ValueError("gamma_max must be >= gamma_min")
        self.gamma_min, self.gamma_max, self.eps = float(gamma_min), float(gamma_max), float(eps)
        self.gammas: dict[int, list[float]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.gammas[index] = (self.gamma_min + torch.rand(len(shapes)) * (self.gamma_max - self.gamma_min)).tolist()
        return shapes

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        flat = tensor.float().reshape(tensor.shape[0], -1)
        low = flat.min(1).values.reshape(-1, *([1] * (tensor.ndim - 1)))
        high = flat.max(1).values.reshape(-1, *([1] * (tensor.ndim - 1)))
        scale = (high - low).clamp_min(self.eps)
        return (((tensor.float() - low) / scale).clamp(0, 1).pow(self.gammas[index][a]) * scale + low).to(tensor.dtype)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor
