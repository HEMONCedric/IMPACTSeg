"""Foreground-oriented patch sampling for the IMPACTSeg training loader."""

from __future__ import annotations

import math
import random
import threading
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from torch.utils.data import DataLoader, Sampler

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager.groups import Group
from konfai.data.data_manager.samples import DatasetIter
from konfai.data.data_manager.sources import DataTrain as KonfAIDataTrain
from konfai.data.data_manager.subset import Subset
from konfai.data.patching import DatasetPatch
from konfai.utils.config import config
from .coarse_sampling import ModalitySampling, ModalityCaseForegroundSampler, EpochPatchReadOrder, case_modalities


@config("foreground_sampling")
class ForegroundSampling:
    """Settings for the foreground/uniform mixture of training patches."""

    def __init__(
        self,
        enabled: bool = True,
        foreground_probability: float = 0.70,
        mask_group: str = "mask_fine",
    ) -> None:
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0, 1].")
        if not mask_group:
            raise ValueError("mask_group must name the segmentation target group.")
        self.enabled = bool(enabled)
        self.foreground_probability = float(foreground_probability)
        self.mask_group = str(mask_group)


@config("banked_streaming")
class BankedStreaming:
    """Settings for the v1-style patient-bank cache in streaming mode."""

    def __init__(
        self,
        enabled: bool = False,
        bank_size: int = 24,
        patients_per_batch: int = 8,
        preload_next_bank: bool = True,
        preload_workers: int = 1,
        seed: int = 42,
    ) -> None:
        if bank_size <= 0:
            raise ValueError("bank_size must be strictly positive.")
        if patients_per_batch <= 0:
            raise ValueError("patients_per_batch must be strictly positive.")
        if preload_workers <= 0:
            raise ValueError("preload_workers must be strictly positive.")
        self.enabled = bool(enabled)
        self.bank_size = int(bank_size)
        self.patients_per_batch = int(patients_per_batch)
        self.preload_next_bank = bool(preload_next_bank)
        self.preload_workers = int(preload_workers)
        self.seed = int(seed)


@config("validation_banked_streaming")
class ValidationBankedStreaming:
    """Settings for deterministic, exhaustive validation patient banks."""

    def __init__(
        self,
        enabled: bool = False,
        bank_size: int = 2,
        preload_next_bank: bool = True,
        preload_workers: int = 2,
    ) -> None:
        if bank_size <= 0:
            raise ValueError("bank_size must be strictly positive.")
        if preload_workers <= 0:
            raise ValueError("preload_workers must be strictly positive.")
        self.enabled = bool(enabled)
        self.bank_size = int(bank_size)
        self.preload_next_bank = bool(preload_next_bank)
        self.preload_workers = int(preload_workers)


class ForegroundUniformSampler(Sampler[int]):
    """Draw a fixed foreground-oriented/uniform mix without touching validation."""

    def __init__(
        self,
        mapping: list[tuple[int, int, int]],
        foreground_indices: list[int],
        foreground_probability: float,
        read_order,
    ) -> None:
        self.mapping = mapping
        self.foreground_indices = torch.as_tensor(foreground_indices, dtype=torch.int64)
        self.foreground_probability = float(foreground_probability)
        self.read_order = read_order

    def __iter__(self) -> Iterator[int]:
        total = len(self.mapping)
        if total == 0:
            return iter(())
        generator = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        foreground_draws = int(round(total * self.foreground_probability)) if len(self.foreground_indices) else 0
        uniform_draws = total - foreground_draws
        pieces: list[torch.Tensor] = []
        if foreground_draws:
            selected = torch.randint(len(self.foreground_indices), (foreground_draws,), generator=generator)
            pieces.append(self.foreground_indices[selected])
        if uniform_draws:
            pieces.append(torch.randint(total, (uniform_draws,), generator=generator))
        order = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64)
        if len(order) > 1:
            order = order[torch.randperm(len(order), generator=generator)]
        self.read_order.publish(order)
        return iter(order.tolist())

    def __len__(self) -> int:
        return len(self.mapping)


class PatientBankForegroundBatchSampler(Sampler[list[int]]):
    """Sample 70/30 first, then consume it from resident banks of patients.

    This ports the useful v1 mechanism to KonfAI 1.8's ``DatasetIter`` API.
    It deliberately runs with ``num_workers=0``: the sampler and ``__getitem__``
    must see the same in-process cache populated by ``_load_data``.
    """

    def __init__(
        self,
        dataset: DatasetIter,
        batch_size: int,
        foreground_indices: list[int],
        foreground_probability: float,
        bank_size: int,
        patients_per_batch: int,
        preload_next_bank: bool,
        seed: int,
        preload_workers: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if preload_workers <= 0:
            raise ValueError("preload_workers must be strictly positive.")
        self.dataset = dataset
        self.mapping = dataset.mapping
        self.batch_size = int(batch_size)
        self.foreground_indices = list(foreground_indices)
        self.foreground_probability = float(foreground_probability)
        self.patients_per_batch = min(max(1, int(patients_per_batch)), self.batch_size)
        self.patches_per_patient = max(1, self.batch_size // self.patients_per_batch)
        self.bank_size = max(self.patients_per_batch, int(bank_size))
        self.preload_next_bank = bool(preload_next_bank)
        self.preload_workers = int(preload_workers)
        self.seed = int(seed)
        self.epoch = 0
        self.case_to_mapping_indices: dict[int, list[int]] = defaultdict(list)
        for mapping_index, (case_index, _, _) in enumerate(self.mapping):
            self.case_to_mapping_indices[int(case_index)].append(mapping_index)
        self._cache_lock = threading.RLock()
        self._loading_cases: set[int] = set()

    def __len__(self) -> int:
        return math.ceil(len(self.mapping) / self.batch_size)

    def _draw_epoch(self, rng: random.Random) -> list[int]:
        total = len(self.mapping)
        if total == 0:
            return []
        if not self.foreground_indices:
            order = list(range(total))
            rng.shuffle(order)
            return order
        foreground_draws = round(total * self.foreground_probability)
        uniform_draws = total - foreground_draws
        order = rng.choices(self.foreground_indices, k=foreground_draws)
        order.extend(rng.randrange(total) for _ in range(uniform_draws))
        rng.shuffle(order)
        return order

    def _bank_plans(self, rng: random.Random) -> list[tuple[list[int], list[int]]]:
        selected = self._draw_epoch(rng)
        selected_by_case: dict[int, list[int]] = defaultdict(list)
        for mapping_index in selected:
            selected_by_case[int(self.mapping[mapping_index][0])].append(mapping_index)

        cases = list(self.case_to_mapping_indices)
        rng.shuffle(cases)
        raw_banks = [cases[start : start + self.bank_size] for start in range(0, len(cases), self.bank_size)]
        raw: list[tuple[list[int], list[int]]] = []
        for bank_cases in raw_banks:
            indices = [index for case in bank_cases for index in selected_by_case.get(case, [])]
            if indices:
                raw.append((bank_cases, indices))
        if not raw:
            return []

        # Every bank must end on a full batch before it can be unloaded. Allocate
        # exactly ceil(epoch_samples / batch_size) batches across banks with the
        # largest-remainder method, then minimally trim/pad each bank's draw.
        target_batches = len(self)
        exact = [len(indices) / self.batch_size for _, indices in raw]
        allocation = [math.floor(value) for value in exact]
        remaining = target_batches - sum(allocation)
        for bank_index in sorted(range(len(raw)), key=lambda i: exact[i] - allocation[i], reverse=True)[:remaining]:
            allocation[bank_index] += 1

        plans: list[tuple[list[int], list[int]]] = []
        for (bank_cases, indices), batches in zip(raw, allocation, strict=True):
            if batches == 0:
                continue
            target = batches * self.batch_size
            rng.shuffle(indices)
            if len(indices) > target:
                indices = indices[:target]
            elif len(indices) < target:
                indices.extend(rng.choices(indices, k=target - len(indices)))
            plans.append((bank_cases, indices))
        if sum(len(indices) for _, indices in plans) != target_batches * self.batch_size:
            raise RuntimeError("Internal patient-bank allocation did not preserve the epoch length.")
        return plans

    def _load_case(self, case_index: int) -> None:
        # Loading different cases concurrently is supported by KonfAI's native
        # cache path.  Reserve the case under our lock, but do the expensive I/O
        # and transforms outside it so preload_workers can actually overlap.
        with self._cache_lock:
            if case_index in self.dataset._index_cache_lookup:
                return
            if case_index in self._loading_cases:
                raise RuntimeError(f"Patient {case_index} was scheduled for loading more than once.")
            self._loading_cases.add(case_index)
        try:
            self.dataset._load_data(case_index)
        finally:
            with self._cache_lock:
                self._loading_cases.remove(case_index)

    def _load_bank(self, cases: list[int]) -> None:
        for case_index in cases:
            self._load_case(case_index)

    def _unload_bank(self, cases: list[int]) -> None:
        with self._cache_lock:
            for case_index in cases:
                if case_index in self.dataset._index_cache_lookup:
                    self.dataset._unload_data(case_index)

    def _reset_cache(self) -> None:
        with self._cache_lock:
            for case_index in list(self.dataset._index_cache):
                self.dataset._unload_data(case_index)

    def _batches(self, indices: list[int], rng: random.Random) -> Iterator[list[int]]:
        queues: dict[int, list[int]] = defaultdict(list)
        for mapping_index in indices:
            queues[int(self.mapping[mapping_index][0])].append(mapping_index)
        for queue in queues.values():
            rng.shuffle(queue)

        while queues:
            available = [case for case, queue in queues.items() if queue]
            rng.shuffle(available)
            selected_cases = available[: self.patients_per_batch]
            batch: list[int] = []
            for case in selected_cases:
                take = min(self.patches_per_patient, len(queues[case]), self.batch_size - len(batch))
                batch.extend(queues[case][:take])
                del queues[case][:take]
            while len(batch) < self.batch_size:
                made_progress = False
                for case in available:
                    if queues[case]:
                        batch.append(queues[case].pop())
                        made_progress = True
                        if len(batch) == self.batch_size:
                            break
                if not made_progress:
                    raise RuntimeError("Patient bank ran out before completing a planned batch.")
            queues = defaultdict(list, {case: queue for case, queue in queues.items() if queue})
            yield batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        plans = self._bank_plans(rng)
        if not plans:
            return iter(())
        self._reset_cache()
        executor: ThreadPoolExecutor | None = None
        futures: list[Future[None]] = []
        preloaded: list[int] | None = None

        def iterate() -> Iterator[list[int]]:
            nonlocal executor, futures, preloaded

            def submit_bank(cases: list[int]) -> list[Future[None]]:
                if executor is None:
                    raise RuntimeError("The patient-bank preload executor is not available.")
                return [executor.submit(self._load_case, case_index) for case_index in cases]

            def wait_bank(pending: list[Future[None]]) -> None:
                try:
                    for pending_future in pending:
                        pending_future.result()
                except BaseException:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise

            try:
                if self.preload_next_bank:
                    executor = ThreadPoolExecutor(
                        max_workers=self.preload_workers,
                        thread_name_prefix="impactseg-bank",
                    )
                    # Parallelise the cold first bank as well; subsequent banks
                    # are loaded by the same workers while the GPU consumes the
                    # current bank.
                    wait_bank(submit_bank(plans[0][0]))
                else:
                    self._load_bank(plans[0][0])
                if executor is not None and len(plans) > 1:
                    preloaded = plans[1][0]
                    futures = submit_bank(preloaded)
                for bank_index, (bank_cases, indices) in enumerate(plans):
                    if bank_index > 0:
                        if preloaded == bank_cases and futures:
                            pending, futures = futures, []
                            wait_bank(pending)
                        else:
                            self._load_bank(bank_cases)
                    next_cases = plans[bank_index + 1][0] if bank_index + 1 < len(plans) else None
                    if self.preload_next_bank and next_cases is not None and preloaded != next_cases:
                        if executor is None:
                            raise RuntimeError("The patient-bank preload executor is not available.")
                        preloaded = next_cases
                        futures = submit_bank(next_cases)
                    yield from self._batches(indices, rng)
                    self._unload_bank(bank_cases)
            finally:
                if futures:
                    pending, futures = futures, []
                    wait_bank(pending)
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                self._reset_cache()

        return iterate()


class SequentialValidationBankBatchSampler(Sampler[list[int]]):
    """Visit every validation patch once using deterministic patient banks.

    Only the active bank and, when enabled, one prefetched reserve bank are
    resident.  A bank is exhausted before its patients are released.
    """

    def __init__(
        self,
        dataset: DatasetIter,
        batch_size: int,
        bank_size: int = 2,
        preload_next_bank: bool = True,
        preload_workers: int = 2,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if bank_size <= 0:
            raise ValueError("bank_size must be strictly positive.")
        if preload_workers <= 0:
            raise ValueError("preload_workers must be strictly positive.")
        self.dataset = dataset
        self.mapping = dataset.mapping
        self.batch_size = int(batch_size)
        self.bank_size = int(bank_size)
        self.preload_next_bank = bool(preload_next_bank)
        self.preload_workers = int(preload_workers)
        self.case_to_mapping_indices: dict[int, list[int]] = defaultdict(list)
        for mapping_index, (case_index, _, _) in enumerate(self.mapping):
            self.case_to_mapping_indices[int(case_index)].append(mapping_index)
        self.cases = list(self.case_to_mapping_indices)
        self._cache_lock = threading.RLock()
        self._loading_cases: set[int] = set()

    def _plans(self) -> list[tuple[list[int], list[int]]]:
        plans: list[tuple[list[int], list[int]]] = []
        for start in range(0, len(self.cases), self.bank_size):
            bank_cases = self.cases[start : start + self.bank_size]
            indices = [index for case in bank_cases for index in self.case_to_mapping_indices[case]]
            if indices:
                plans.append((bank_cases, indices))
        return plans

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for _, indices in self._plans())

    def _load_case(self, case_index: int) -> None:
        with self._cache_lock:
            if case_index in self.dataset._index_cache_lookup:
                return
            if case_index in self._loading_cases:
                raise RuntimeError(f"Validation patient {case_index} was scheduled for loading more than once.")
            self._loading_cases.add(case_index)
        try:
            self.dataset._load_data(case_index)
        finally:
            with self._cache_lock:
                self._loading_cases.remove(case_index)

    def _load_bank(self, cases: list[int]) -> None:
        for case_index in cases:
            self._load_case(case_index)

    def _unload_bank(self, cases: list[int]) -> None:
        with self._cache_lock:
            for case_index in cases:
                if case_index in self.dataset._index_cache_lookup:
                    self.dataset._unload_data(case_index)

    def _reset_cache(self) -> None:
        with self._cache_lock:
            for case_index in list(self.dataset._index_cache):
                self.dataset._unload_data(case_index)

    def __iter__(self) -> Iterator[list[int]]:
        plans = self._plans()
        if not plans:
            return iter(())

        # Keep KonfAI's patch-read bookkeeping aligned with the exact,
        # deterministic order imposed by this batch sampler.
        order = [index for _, indices in plans for index in indices]
        self.dataset.read_order.publish(torch.as_tensor(order, dtype=torch.int64))
        self._reset_cache()
        executor: ThreadPoolExecutor | None = None
        futures: list[Future[None]] = []
        preloaded: list[int] | None = None

        def iterate() -> Iterator[list[int]]:
            nonlocal executor, futures, preloaded

            def submit_bank(cases: list[int]) -> list[Future[None]]:
                if executor is None:
                    raise RuntimeError("The validation-bank preload executor is not available.")
                return [executor.submit(self._load_case, case_index) for case_index in cases]

            def wait_bank(pending: list[Future[None]]) -> None:
                try:
                    for pending_future in pending:
                        pending_future.result()
                except BaseException:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise

            try:
                if self.preload_next_bank:
                    executor = ThreadPoolExecutor(
                        max_workers=self.preload_workers,
                        thread_name_prefix="impactseg-validation-bank",
                    )
                    wait_bank(submit_bank(plans[0][0]))
                else:
                    self._load_bank(plans[0][0])

                if executor is not None and len(plans) > 1:
                    preloaded = plans[1][0]
                    futures = submit_bank(preloaded)

                for bank_index, (bank_cases, indices) in enumerate(plans):
                    if bank_index > 0:
                        if preloaded == bank_cases and futures:
                            pending, futures = futures, []
                            wait_bank(pending)
                        else:
                            self._load_bank(bank_cases)

                    next_cases = plans[bank_index + 1][0] if bank_index + 1 < len(plans) else None
                    if self.preload_next_bank and next_cases is not None and preloaded != next_cases:
                        if executor is None:
                            raise RuntimeError("The validation-bank preload executor is not available.")
                        preloaded = next_cases
                        futures = submit_bank(next_cases)

                    for start in range(0, len(indices), self.batch_size):
                        yield indices[start : start + self.batch_size]
                    self._unload_bank(bank_cases)
            finally:
                if futures:
                    pending, futures = futures, []
                    wait_bank(pending)
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                self._reset_cache()

        return iterate()


def _banked_loader(train_loader: DataLoader, batch_sampler: Sampler[list[int]]) -> DataLoader:
    """Rebuild a KonfAI loader around an in-process patient-bank sampler."""

    kwargs = {
        "dataset": train_loader.dataset,
        "batch_sampler": batch_sampler,
        "num_workers": 0,
        "collate_fn": train_loader.collate_fn,
        "pin_memory": train_loader.pin_memory,
        "timeout": train_loader.timeout,
        "worker_init_fn": train_loader.worker_init_fn,
        "generator": train_loader.generator,
    }
    pin_memory_device = getattr(train_loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(**kwargs)


def _foreground_patch_indices(dataset: DatasetIter, mask_group: str, keep_loaded: bool) -> list[int]:
    """Return mapping positions whose transformed target patch contains a non-zero ID."""

    if mask_group not in dataset.data:
        available = ", ".join(sorted(dataset.data))
        raise RuntimeError(f"Foreground sampling target '{mask_group}' is unavailable (groups: {available}).")

    managers = dataset.data[mask_group]
    loaded_here: list[object] = []
    for manager in managers:
        if not manager.loaded:
            manager.load(manager.transforms, dataset.data_augmentations_list, load_augmentations=True)
            loaded_here.append(manager)

    foreground: list[int] = []
    try:
        for mapping_index, (case_index, augmentation_index, patch_index) in enumerate(dataset.mapping):
            manager = managers[case_index]
            tensor = manager._get_tensor(augmentation_index)
            spatial_slices = manager.patch.get_patch_slices(augmentation_index)[patch_index]
            if torch.any(tensor[(slice(None), *spatial_slices)] != 0):
                foreground.append(mapping_index)
    finally:
        if not keep_loaded:
            for manager in loaded_here:
                manager.unload()
    return foreground


@config("Dataset")
class ForegroundSamplingDataTrain(KonfAIDataTrain):
    """KonfAI DataTrain with an optional sampler on the training loader only."""

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, Group] = {"default|Labels": Group()},
        augmentations: dict[str, DataAugmentationsList] | None = {"DataAugmentation_0": DataAugmentationsList()},
        inline_augmentations: bool = False,
        patch: DatasetPatch | None = DatasetPatch(),
        memory_budget: str | float | None = None,
        subset: Subset = Subset(),
        batch_size: int = 1,
        validation: float | str | list[int] | list[str] | None = 0.2,
        validation_augmentations: bool = True,
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
        foreground_sampling: ForegroundSampling | None = None,
        banked_streaming: BankedStreaming | None = None,
        validation_banked_streaming: ValidationBankedStreaming | None = None,
        modality_sampling: ModalitySampling | None = None,
    ) -> None:
        super().__init__(
            dataset_filenames=dataset_filenames,
            groups_src=groups_src,
            augmentations=augmentations,
            inline_augmentations=inline_augmentations,
            patch=patch,
            memory_budget=memory_budget,
            subset=subset,
            batch_size=batch_size,
            validation=validation,
            validation_augmentations=validation_augmentations,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.foreground_sampling = foreground_sampling
        self.banked_streaming = banked_streaming
        self.validation_banked_streaming = validation_banked_streaming
        self.modality_sampling = modality_sampling

    def get_data(self, world_size: int):
        data_loaders, case_names, validation_names = super().get_data(world_size)
        validation_banked = self.validation_banked_streaming
        if validation_banked is not None and validation_banked.enabled and not self.use_cache:
            for rank, loaders in enumerate(data_loaders):
                if len(loaders) < 2:
                    continue
                validation_loader = loaders[1]
                sampler = SequentialValidationBankBatchSampler(
                    dataset=validation_loader.dataset,
                    batch_size=self.batch_size,
                    bank_size=validation_banked.bank_size,
                    preload_next_bank=validation_banked.preload_next_bank,
                    preload_workers=validation_banked.preload_workers,
                )
                reserve_banks = 1 if sampler.preload_next_bank else 0
                print(
                    f"[IMPACTSeg] sequential validation-bank streaming rank {rank}: "
                    f"bank={sampler.bank_size}, reserve_banks={reserve_banks}, "
                    f"preload_workers={sampler.preload_workers}, shuffle=False, "
                    f"exhaustive=True, dataloader_workers=0."
                )
                loaders[1] = _banked_loader(validation_loader, sampler)

        settings = self.foreground_sampling
        balanced = self.modality_sampling
        if (settings is None or not settings.enabled) and not (balanced and balanced.enabled):
            return data_loaders, case_names, validation_names

        for rank, loaders in enumerate(data_loaders):
            train_loader = loaders[0]
            foreground = _foreground_patch_indices(
                train_loader.dataset,
                settings.mask_group,
                keep_loaded=self.use_cache,
            ) if settings is not None and settings.enabled else []
            if balanced is not None and balanced.enabled:
                if world_size != 1 or (self.banked_streaming is not None and self.banked_streaming.enabled):
                    raise ValueError("Modality-balanced coarse sampling currently requires one GPU and no banking")
                train_loader.dataset.read_order = EpochPatchReadOrder(
                    train_loader.dataset.mapping, self.batch_size, balanced.samples_per_epoch,
                )
                sampler = ModalityCaseForegroundSampler(
                    train_loader.dataset.mapping, case_modalities(train_loader.dataset, balanced), foreground,
                    settings.foreground_probability if settings and settings.enabled else 0.0,
                    balanced.samples_per_epoch, balanced.seed + rank, train_loader.dataset.read_order,
                )
                loaders[0] = DataLoader(dataset=train_loader.dataset, sampler=sampler,
                                       batch_size=self.batch_size, **self.dataLoader_args)
                print("[IMPACTSeg] coarse modality sampling: CT/CBCT/MRI=1/3 each; uniform cases; "
                      f"{balanced.samples_per_epoch} patches/epoch; validation unchanged.")
                continue
            if not foreground:
                print(f"[IMPACTSeg] foreground sampling rank {rank}: no foreground patch found; using uniform draws.")
            else:
                print(
                    f"[IMPACTSeg] foreground sampling rank {rank}: {len(foreground)}/{len(train_loader.dataset)} "
                    f"eligible patches; {settings.foreground_probability:.0%} foreground-oriented / "
                    f"{1.0 - settings.foreground_probability:.0%} uniform draws."
                )
            banked = self.banked_streaming
            if banked is not None and banked.enabled and not self.use_cache:
                sampler = PatientBankForegroundBatchSampler(
                    dataset=train_loader.dataset,
                    batch_size=self.batch_size,
                    foreground_indices=foreground,
                    foreground_probability=settings.foreground_probability,
                    bank_size=banked.bank_size,
                    patients_per_batch=banked.patients_per_batch,
                    preload_next_bank=banked.preload_next_bank,
                    seed=banked.seed + rank,
                    preload_workers=banked.preload_workers,
                )
                print(
                    f"[IMPACTSeg] v1-style patient-bank streaming rank {rank}: "
                    f"bank={sampler.bank_size}, patients/batch={sampler.patients_per_batch}, "
                    f"patches/patient~={sampler.patches_per_patient}, preload={sampler.preload_next_bank}, "
                    f"preload_workers={sampler.preload_workers}, dataloader_workers=0."
                )
                loaders[0] = _banked_loader(train_loader, sampler)
                continue
            loaders[0] = DataLoader(
                dataset=train_loader.dataset,
                sampler=ForegroundUniformSampler(
                    train_loader.dataset.mapping,
                    foreground,
                    settings.foreground_probability,
                    train_loader.dataset.read_order,
                ),
                batch_size=self.batch_size,
                **self.dataLoader_args,
            )
        return data_loaders, case_names, validation_names
