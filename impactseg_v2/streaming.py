"""Compatibility fixes for KonfAI patch-native streaming."""

from __future__ import annotations


def install_extend_slice_region_streaming_fix() -> None:
    """Make KonfAI region streaming include the configured 2.5D slice context.

    KonfAI 1.8.3 builds the regional replay target from ``read_slices``. That
    contains the core patch but not ``extend_slice``. Its finalizer then tries
    to concatenate context which was never read, collapsing a ``[1, H, W]``
    core to ``[H, W]``. Use the authoritative read plan as the replay target
    for input groups with extended slices.
    """

    from konfai.data.patching.manager import DatasetManager
    from konfai.utils.dataset import Attribute

    if getattr(DatasetManager, "_impactseg_extend_slice_region_fix", False):
        return

    original = DatasetManager._get_streamed_region_data

    def patched(self, index, a, stream_source, is_input):
        if not is_input or self.patch is None or self.patch.extend_slice <= 0:
            return original(self, index, a, stream_source, is_input)

        if self._expand is not None and self._records_source is not stream_source:
            self._refold_copy_records(a, stream_source)

        plan = self.patch.get_read_plan(self.shapes[a], index, a, is_input)
        spatial_rank = len(self.patch.get_patch_slices(a)[index])
        target_slices = tuple(plan.data_slices[-spatial_rank:])
        persist = a not in self._stream_attributes_persisted
        tensor, cache_attribute, keys_before = self._replay_streamed_region(
            stream_source,
            target_slices,
            Attribute(self.cache_attributes_bak[a]),
            self.cache_attributes[a] if persist else None,
        )
        tensor = self._finalize_stream_patch(tensor, index, a, is_input)
        if persist:
            self._persist_stream_attributes(a, cache_attribute, keys_before)
        return tensor, cache_attribute

    DatasetManager._get_streamed_region_data = patched
    DatasetManager._impactseg_extend_slice_region_fix = True
