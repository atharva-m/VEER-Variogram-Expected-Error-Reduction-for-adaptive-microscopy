"""SEM-EDS ingestion and standardized observable dataset contracts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from scipy.ndimage import zoom

from .domain import DatasetConfig, ReplayCapabilities, RunConfig, SpectrumWindow


def _add_contract_fields(dataset: xr.Dataset) -> xr.Dataset:
    observations = int(dataset.sizes["observation"])
    result = dataset.copy()
    if "acquisition_id" not in result.coords:
        result = result.assign_coords(
            acquisition_id=("observation", np.full(observations, "source", dtype="U32"))
        )
    if "valid_observation" not in result:
        result["valid_observation"] = ("observation", np.ones(observations, dtype=bool))
    if "y_step_nm" not in result.coords and "step_nm" in result.coords:
        result = result.assign_coords(y_step_nm=result["step_nm"])
    required = ("x_nm", "y_nm", "step_nm", "dwell_ms")
    missing = [name for name in required if name not in result.coords]
    if missing:
        raise ValueError(f"standardized elemental observations are missing coordinates: {missing}")
    if "counts" not in result or result["counts"].dims != ("observation", "element"):
        raise ValueError("standardized data must provide counts[observation, element]")
    return result


def dense_counts_to_observations(
    counts: xr.DataArray,
    dwell_ms: float,
    step_nm: float,
    y_step_nm: float | None = None,
    acquisition_id: str = "dense_source",
    valid_region: xr.DataArray | None = None,
) -> xr.Dataset:
    """Flatten dense counts[element, y, x] into the observable replay contract."""

    if set(counts.dims) != {"element", "y", "x"}:
        raise ValueError("dense count maps must have dimensions element, y, x")
    ordered = counts.transpose("element", "y", "x")
    y_step_nm = y_step_nm or step_nm
    x = ordered.coords["x"].values.astype(float)
    y = ordered.coords["y"].values.astype(float)
    xx, yy = np.meshgrid(x, y)
    n = xx.size
    dataset = _add_contract_fields(
        xr.Dataset(
            {"counts": (("observation", "element"), ordered.values.reshape(ordered.sizes["element"], n).T)},
            coords={
                "observation": np.arange(n),
                "element": ordered.coords["element"].values.astype(str),
                "x_nm": ("observation", xx.ravel()),
                "y_nm": ("observation", yy.ravel()),
                "step_nm": ("observation", np.full(n, step_nm)),
                "y_step_nm": ("observation", np.full(n, y_step_nm)),
                "dwell_ms": ("observation", np.full(n, dwell_ms)),
                "acquisition_id": ("observation", np.full(n, acquisition_id, dtype="U32")),
            },
        )
    )
    if valid_region is not None:
        if set(valid_region.dims) != {"y", "x"}:
            raise ValueError("dense valid-region masks must have dimensions y, x")
        dataset["valid_observation"] = (
            "observation",
            valid_region.transpose("y", "x").values.astype(bool).ravel(),
        )
    return dataset


def extract_element_counts(
    spectrum: np.ndarray,
    energies: np.ndarray,
    windows: dict[str, SpectrumWindow],
) -> np.ndarray:
    """Extract background-corrected element counts using configured energy windows."""

    results = []
    for element, window in windows.items():
        peak = (energies >= window.peak_range[0]) & (energies <= window.peak_range[1])
        if not np.any(peak):
            raise ValueError(f"no energy channels fall within the peak window for {element}")
        peak_counts = spectrum[:, peak].sum(axis=1).astype(float)
        background_counts = np.zeros(spectrum.shape[0], dtype=float)
        background_channels = 0
        for low, high in window.background_ranges:
            mask = (energies >= low) & (energies <= high)
            background_counts += spectrum[:, mask].sum(axis=1)
            background_channels += int(mask.sum())
        if background_channels:
            peak_counts -= background_counts * int(peak.sum()) / background_channels
        results.append(np.maximum(np.rint(peak_counts), 0).astype(np.int64))
    return np.stack(results, axis=1)


def _generic_dataset(config: RunConfig) -> xr.Dataset:
    dataset_config = config.dataset
    if dataset_config.source is None:
        raise ValueError("generic element-map ingestion requires dataset.source")
    source = Path(dataset_config.source)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        required = {"x_nm", "y_nm", "dwell_ms", "step_nm"}
        if not required.issubset(frame.columns):
            raise ValueError("CSV element maps require x_nm, y_nm, dwell_ms, and step_nm columns")
        if {"element", "counts"}.issubset(frame.columns):
            index = ["x_nm", "y_nm", "dwell_ms", "step_nm"]
            wide = frame.pivot_table(index=index, columns="element", values="counts", aggfunc="sum").reset_index()
        else:
            wide = frame
        if not set(config.scenario.elements).issubset(wide.columns):
            raise ValueError("CSV count maps must contain every configured element channel")
        counts = wide[config.scenario.elements].to_numpy(dtype=np.int64)
        n = len(wide)
        return _add_contract_fields(
            xr.Dataset(
                {"counts": (("observation", "element"), counts)},
                coords={
                    "observation": np.arange(n),
                    "element": config.scenario.elements,
                    "x_nm": ("observation", wide["x_nm"].to_numpy(float)),
                    "y_nm": ("observation", wide["y_nm"].to_numpy(float)),
                    "step_nm": ("observation", wide["step_nm"].to_numpy(float)),
                    "dwell_ms": ("observation", wide["dwell_ms"].to_numpy(float)),
                },
            )
        )
    dataset = xr.open_zarr(source) if source.suffix.lower() == ".zarr" or source.is_dir() else xr.open_dataset(source)
    if "counts" in dataset and dataset["counts"].dims == ("observation", "element"):
        return _add_contract_fields(dataset.load())
    variable = "counts" if "counts" in dataset else "count_map"
    if variable not in dataset:
        raise ValueError("generic map dataset must contain counts or count_map")
    dwell = dataset_config.dwell_ms or float(dataset.attrs.get("dwell_ms", config.instrument.fine_dwell_ms))
    step = dataset_config.x_step_nm or float(dataset.attrs.get("step_nm", config.instrument.fine_step_nm))
    validity = None
    for name in ("valid_region", "valid_pixel", "valid_observation"):
        if name in dataset and set(dataset[name].dims) == {"y", "x"}:
            validity = dataset[name].load()
            break
    y_step = dataset_config.y_step_nm or float(dataset.attrs.get("y_step_nm", step))
    return dense_counts_to_observations(
        dataset[variable].load(), dwell, step, y_step_nm=y_step, valid_region=validity
    )


def _read_map_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2)
    if values.ndim != 2:
        raise ValueError(f"element-map images must be two dimensional: {path}")
    return np.maximum(np.rint(values), 0).astype(np.int64)


def _crop_map(values: np.ndarray, crop: tuple[int, int, int, int], name: str) -> np.ndarray:
    row0, row1, column0, column1 = crop
    if row1 > values.shape[0] or column1 > values.shape[1]:
        raise ValueError(f"configured map crop for {name} exceeds source image dimensions")
    return values[row0:row1, column0:column1]


def _aligned_map_dataset(
    config: RunConfig, raw: dict[str, np.ndarray], source_adapter: str
) -> xr.Dataset:
    """Standardize dense per-element maps with explicit, auditable alignment."""
    dataset_config = config.dataset
    alignment = dataset_config.map_alignment
    if set(dataset_config.element_map_sources) != set(config.scenario.elements):
        raise ValueError("element-map sources must provide every configured element")
    original_shapes = {element: list(values.shape) for element, values in raw.items()}
    maps = {
        element: _crop_map(values, alignment.crops[element], element)
        if element in alignment.crops
        else values
        for element, values in raw.items()
    }
    reference = alignment.reference_element or config.scenario.elements[0]
    if reference not in maps:
        raise ValueError("map alignment reference_element must name a configured element")
    reference_shape = maps[reference].shape
    aligned = {}
    for element in config.scenario.elements:
        values = maps[element]
        if values.shape == reference_shape:
            aligned[element] = values
            continue
        if alignment.method in ("strict", "configured_crop"):
            raise ValueError(
                f"element map {element} shape {values.shape} does not match reference "
                f"{reference} shape {reference_shape}; supply map_alignment.crops or "
                "declare resample_to_reference with shared physical extent"
            )
        order = 0 if alignment.interpolation == "nearest" else 1
        factors = (
            reference_shape[0] / values.shape[0],
            reference_shape[1] / values.shape[1],
        )
        aligned[element] = np.maximum(np.rint(zoom(values, factors, order=order)), 0).astype(np.int64)
    if dataset_config.spatial_crop_indices is not None:
        for element, values in aligned.items():
            aligned[element] = _crop_map(
                values, dataset_config.spatial_crop_indices, f"{element} analysis ROI"
            )
    shape = aligned[reference].shape
    if any(values.shape != shape for values in aligned.values()):
        raise RuntimeError("element-map alignment failed to produce a common grid")
    step_x = dataset_config.x_step_nm or config.instrument.fine_step_nm
    step_y = dataset_config.y_step_nm or step_x
    dwell = dataset_config.dwell_ms or config.instrument.fine_dwell_ms
    x = (np.arange(shape[1]) + 0.5) * step_x
    y = (np.arange(shape[0]) + 0.5) * step_y
    count_map = xr.DataArray(
        np.stack([aligned[element] for element in config.scenario.elements]),
        dims=("element", "y", "x"),
        coords={"element": config.scenario.elements, "x": x, "y": y},
    )
    result = dense_counts_to_observations(
        count_map, dwell, step_x, y_step_nm=step_y, acquisition_id="image_source"
    )
    result.attrs.update(
        {
            "source_adapter": source_adapter,
            "value_semantics": dataset_config.value_semantics,
            "alignment_method": alignment.method,
            "reference_element": reference,
            "original_shapes": original_shapes,
            "aligned_shape": list(shape),
            "spatial_crop_indices": list(dataset_config.spatial_crop_indices)
            if dataset_config.spatial_crop_indices is not None
            else None,
            "source_paths": {
                element: str(path) for element, path in dataset_config.element_map_sources.items()
            },
        }
    )
    return result


def _image_map_dataset(config: RunConfig) -> xr.Dataset:
    """Load elemental TIFF/PNG maps with explicit, auditable spatial alignment."""

    raw = {
        element: _read_map_image(Path(path))
        for element, path in config.dataset.element_map_sources.items()
    }
    return _aligned_map_dataset(config, raw, "element_map_images")


def _read_binary_map(config: DatasetConfig, path: Path) -> np.ndarray:
    dtype = {
        "uint32_le": "<u4",
        "uint16_le": "<u2",
        "float32_le": "<f4",
    }[config.binary_dtype]
    values = np.fromfile(path, dtype=dtype)
    if config.binary_dimensions_from_header:
        if values.size < 2:
            raise ValueError(f"binary element map has no dimension header: {path}")
        columns, rows = int(values[0]), int(values[1])
    else:
        rows, columns = config.binary_shape
    offset = config.binary_data_offset_values
    required = offset + rows * columns
    if rows <= 0 or columns <= 0 or required > values.size:
        raise ValueError(f"binary element-map dimensions exceed available values: {path}")
    mapped = values[offset:required].reshape(rows, columns)
    if not np.all(np.isfinite(mapped)):
        raise ValueError(f"binary element map contains non-finite measurements: {path}")
    return np.maximum(np.rint(mapped), 0).astype(np.int64)


def _binary_map_dataset(config: RunConfig) -> xr.Dataset:
    """Load dense binary element maps whose data layout is explicitly configured."""

    raw = {
        element: _read_binary_map(config.dataset, Path(path))
        for element, path in config.dataset.element_map_sources.items()
    }
    return _aligned_map_dataset(config, raw, "binary_element_map")


def _find_largest_numeric_dataset(handle: h5py.File) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, value: object) -> None:
        if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number) and value.ndim >= 2:
            candidates.append((int(np.prod(value.shape)), name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("HDF5 file contains no numeric spectral-image candidate dataset")
    return max(candidates)[1]


def _ornl_h5_dataset(config: RunConfig) -> xr.Dataset:
    dataset_config = config.dataset
    if dataset_config.source is None:
        raise ValueError("ORNL HDF5 ingestion requires dataset.source")
    if not dataset_config.spectral_windows:
        raise ValueError("ORNL HDF5 ingestion requires configured spectral element windows")
    with h5py.File(dataset_config.source, "r") as handle:
        dataset_path = dataset_config.dataset_path or _find_largest_numeric_dataset(handle)
        raw = handle[dataset_path]
        original_shape = raw.shape
        if dataset_config.energy_path:
            energies = np.asarray(handle[dataset_config.energy_path]).ravel()
        else:
            energies = np.arange(raw.shape[-1], dtype=float)
        if raw.ndim == 3:
            source_rows, source_columns = raw.shape[:2]
        elif dataset_config.grid_shape is not None:
            source_rows, source_columns = dataset_config.grid_shape
        else:
            raise ValueError("flattened spectral HDF5 data require dataset.grid_shape")
        if source_rows * source_columns != int(np.prod(raw.shape[:-1])):
            raise ValueError("configured HDF5 grid shape does not match spectrum count")
        if dataset_config.spatial_crop_indices is None:
            row0, row1, column0, column1 = 0, source_rows, 0, source_columns
        else:
            row0, row1, column0, column1 = dataset_config.spatial_crop_indices
            if row1 > source_rows or column1 > source_columns:
                raise ValueError("HDF5 spatial crop extends beyond source grid dimensions")
        rows, columns = row1 - row0, column1 - column0
        if raw.ndim == 3:
            spectra = np.asarray(raw[row0:row1, column0:column1, :]).reshape(-1, raw.shape[-1])
        else:
            spectra = np.concatenate(
                [
                    np.asarray(raw[row * source_columns + column0 : row * source_columns + column1, :])
                    for row in range(row0, row1)
                ],
                axis=0,
            )
    if energies.size != spectra.shape[1]:
        raise ValueError("energy axis length does not match spectral channels")
    step_x = dataset_config.x_step_nm or config.instrument.fine_step_nm
    step_y = dataset_config.y_step_nm or step_x
    dwell = dataset_config.dwell_ms or config.instrument.fine_dwell_ms
    xs = (np.arange(columns) + 0.5) * step_x
    ys = (np.arange(rows) + 0.5) * step_y
    xx, yy = np.meshgrid(xs, ys)
    counts = extract_element_counts(spectra, energies, dataset_config.spectral_windows)
    variables = {"counts": (("observation", "element"), counts)}
    if dataset_config.retain_spectrum:
        variables["spectrum"] = (("observation", "energy"), spectra)
    result = xr.Dataset(
        variables,
        coords={
            "observation": np.arange(spectra.shape[0]),
            "element": list(dataset_config.spectral_windows),
            "energy": energies,
            "x_nm": ("observation", xx.ravel()),
            "y_nm": ("observation", yy.ravel()),
            "step_nm": ("observation", np.full(spectra.shape[0], step_x)),
            "y_step_nm": ("observation", np.full(spectra.shape[0], step_y)),
            "dwell_ms": ("observation", np.full(spectra.shape[0], dwell)),
        },
        attrs={
            "source_adapter": "ornl_usid_h5",
            "hdf5_dataset_path": dataset_path,
            "source_grid_shape": (source_rows, source_columns),
            "spatial_crop_indices": (row0, row1, column0, column1),
        },
    )
    return _add_contract_fields(result)


def ingest_dataset(config: RunConfig) -> tuple[xr.Dataset, ReplayCapabilities]:
    """Load and standardize the configured real-data source."""

    adapter = config.dataset.adapter
    if adapter in ("generic_element_map", "standardized_zarr"):
        dataset = _generic_dataset(config)
    elif adapter == "element_map_images":
        dataset = _image_map_dataset(config)
    elif adapter == "binary_element_map":
        dataset = _binary_map_dataset(config)
    elif adapter == "ornl_usid_h5":
        dataset = _ornl_h5_dataset(config)
    else:
        raise ValueError(f"unsupported dataset adapter: {adapter}")
    if set(dataset.coords["element"].values.astype(str)) != set(config.scenario.elements):
        raise ValueError("ingested dataset element channels do not match the scenario configuration")
    capabilities = config.dataset.capabilities
    inferred = {
        "native_step_nm": capabilities.native_step_nm or float(dataset["step_nm"].min()),
        "native_dwell_ms": capabilities.native_dwell_ms or float(dataset["dwell_ms"].max()),
        "native_y_step_nm": capabilities.native_y_step_nm or float(dataset["y_step_nm"].min()),
        "available_step_sizes_nm": capabilities.available_step_sizes_nm
        or [float(dataset["step_nm"].min())],
        "available_dwell_times_ms": capabilities.available_dwell_times_ms
        or [float(dataset["dwell_ms"].max())],
        "available_y_step_sizes_nm": capabilities.available_y_step_sizes_nm
        or [float(dataset["y_step_nm"].min())],
    }
    return dataset, capabilities.model_copy(update=inferred)
