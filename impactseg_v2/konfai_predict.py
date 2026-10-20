"""KonfAI PREDICTION entrypoint with the resident ensemble optimization."""

from __future__ import annotations

import runpy

from impactseg_v2.ensemble import install_resident_ensemble


def main() -> None:
    install_resident_ensemble()
    runpy.run_module("konfai.main", run_name="__main__")


if __name__ == "__main__":
    main()
