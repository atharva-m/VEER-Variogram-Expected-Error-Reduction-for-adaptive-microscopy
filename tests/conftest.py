from pathlib import Path

import pytest

from veer.io import load_config


@pytest.fixture()
def small_config():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "alloy617_veer.yaml")
    scenario = config.scenario.model_copy(
        update={"width_nm": 1600.0, "height_nm": 1600.0, "grid_size": 16}
    )
    instrument = config.instrument.model_copy(update={"tile_size_nm": 400.0})
    return config.model_copy(update={"scenario": scenario, "instrument": instrument})
