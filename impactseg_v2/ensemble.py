"""Memory-resident KonfAI checkpoint ensemble for inference."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from konfai.network.network import Network
from konfai.predictor.ensemble import ModelComposite, _colocate_loaded_modules
from konfai.utils.errors import PredictorError


class ResidentModelComposite(ModelComposite):
    """Keep every checkpoint model resident instead of swapping weights per batch."""

    def load(self, state_sources: list[dict[str, Any] | Path | str]) -> None:
        if not state_sources and any(
            parameter.numel() for parameter in self._get_model().parameters()
        ):
            raise PredictorError(
                "ResidentModelComposite received no checkpoint for a parameterised model.",
                "At least one checkpoint is required.",
                "Pass one or more model paths to --models.",
            )

        self._state_sources = state_sources
        self._loaded = True
        self._loaded_state_index = None
        self._state_cache = {}
        base_model = self._get_model()
        resident_models = []
        for index, source in enumerate(state_sources):
            model = base_model if index == 0 else copy.deepcopy(base_model)
            state = self._read_state_source(source)
            model.set_name(self._base_model_name)
            model.load(state, init=False)
            _colocate_loaded_modules(model)
            model.set_name(f"{self._base_model_name}_{index}")
            model.eval()
            resident_models.append(model)
            del state
        self.__dict__["_impactseg_resident_models"] = resident_models
        if len(resident_models) > 1:
            print(
                f"[IMPACTSeg] resident ensemble: {len(resident_models)} models kept on device."
            )

    def _model_for_index(self, index: int):
        models = self.__dict__.get("_impactseg_resident_models")
        if models is None:
            return super()._model_for_index(index)
        model = models[index]
        target_device = next(self._get_model().parameters()).device
        model_device = next(model.parameters()).device
        if target_device.type == "cuda" and model_device != target_device:
            Network.to(model, target_device.index or 0)
            _colocate_loaded_modules(model)
        return model


def install_resident_ensemble() -> None:
    """Patch only the Predictor factory reference; the installed KonfAI stays untouched."""

    import konfai.predictor as predictor
    import konfai.predictor.workflow as workflow

    workflow.ModelComposite = ResidentModelComposite
    predictor.ModelComposite = ResidentModelComposite
