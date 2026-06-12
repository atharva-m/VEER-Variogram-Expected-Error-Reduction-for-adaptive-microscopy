from pathlib import Path

import pytest

from balance_nm.domain import AcquisitionConfig, BenchmarkConfig, Budget, ModelConfig
from balance_nm.io import load_config


@pytest.fixture()
def small_config():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "alloy617_v4_uncertainty.yaml")
    scenario = config.scenario.model_copy(
        update={"width_nm": 1600.0, "height_nm": 1600.0, "grid_size": 16}
    )
    instrument = config.instrument.model_copy(update={"tile_size_nm": 400.0})
    return config.model_copy(
        update={
            "scenario": scenario,
            "instrument": instrument,
            "budget": Budget(max_scan_time_s=4.0, max_dose_proxy=1.0),
            "model": ModelConfig(max_training_points=128, max_observations=512),
            "acquisition": AcquisitionConfig(maximum_follow_on_actions=3),
            "benchmark": BenchmarkConfig(
                seeds=1, policies=["uniform", "gradient", "balance"]
            ),
        }
    )
