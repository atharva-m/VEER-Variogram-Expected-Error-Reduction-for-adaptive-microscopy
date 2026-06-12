"""Optional compact neural ensemble for v4 masked multichannel reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
from scipy.ndimage import distance_transform_edt

from .domain import AcquisitionV4NeuralConfig


def torch_available() -> bool:
    """Return whether the optional learned runtime is installed."""

    return importlib.util.find_spec("torch") is not None


def nearest_signal_and_distance(signal: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the observable nearest-neighbor reconstruction and normalized distance map."""

    observed = np.asarray(observed_mask, dtype=bool)
    rows, columns = observed.shape
    if not np.any(observed):
        return np.zeros_like(signal, dtype=float), np.ones((rows, columns), dtype=float)
    distances, indices = distance_transform_edt(~observed, return_indices=True)
    nearest_y, nearest_x = indices
    nearest = np.asarray(signal, dtype=float)[:, nearest_y, nearest_x]
    normalized_distance = np.clip(distances / max(float(np.hypot(rows, columns)) / 2.0, 1.0), 0.0, 1.0)
    normalized_distance[observed] = 0.0
    return nearest, normalized_distance


def _random_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    rows, columns = shape
    mask = np.zeros(shape, dtype=bool)
    roi_rows = max(2, rows // 4)
    roi_columns = max(2, columns // 4)
    count = int(rng.integers(2, 7))
    for _ in range(count):
        row0 = int(rng.integers(0, max(rows - roi_rows + 1, 1)))
        column0 = int(rng.integers(0, max(columns - roi_columns + 1, 1)))
        mask[row0 : row0 + roi_rows, column0 : column0 + roi_columns] = True
    return mask


def _robust_channel_scale(signals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    channels = signals[0].shape[0]
    lows = np.zeros(channels, dtype=float)
    spans = np.ones(channels, dtype=float)
    for channel in range(channels):
        values = np.concatenate([signal[channel].ravel() for signal in signals])
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.percentile(finite, [1.0, 99.0])
            lows[channel] = float(low)
            spans[channel] = max(float(high - low), 1.0)
    return lows, spans


def _scale(signal: np.ndarray, lows: np.ndarray, spans: np.ndarray) -> np.ndarray:
    return np.clip((signal - lows[:, None, None]) / spans[:, None, None], 0.0, 1.0)


@dataclass
class NeuralReconstructionEnsemble:
    """Small U-Net ensemble loaded only when the optional torch extra is installed."""

    models: list[object]
    lows: np.ndarray
    spans: np.ndarray
    device: str

    def predict(self, visible_signal: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        nearest, distance = nearest_signal_and_distance(visible_signal, observed_mask)
        scaled = _scale(nearest, self.lows, self.spans)
        network_input = np.concatenate(
            [scaled, observed_mask[None].astype(float), distance[None]],
            axis=0,
        )
        tensor = torch.from_numpy(network_input[None].astype(np.float32)).to(self.device)
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predicted = model(tensor).detach().cpu().numpy()[0]
            predictions.append(predicted)
        stacked = np.stack(predictions)
        mean = stacked.mean(axis=0) * self.spans[:, None, None] + self.lows[:, None, None]
        variance = stacked.var(axis=0) * self.spans[:, None, None] ** 2
        return mean, variance


def _build_unet(input_channels: int, output_channels: int, config: AcquisitionV4NeuralConfig):
    import torch
    from torch import nn

    class ConvBlock(nn.Module):
        def __init__(self, first: int, second: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(first, second, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(config.dropout),
                nn.Conv2d(second, second, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, values):
            return self.layers(values)

    class CompactUNet(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [config.base_channels * (2**index) for index in range(config.depth)]
            self.encoders = nn.ModuleList()
            previous = input_channels
            for width in widths:
                self.encoders.append(ConvBlock(previous, width))
                previous = width
            self.pool = nn.MaxPool2d(2)
            self.decoders = nn.ModuleList()
            self.upsamples = nn.ModuleList()
            for width in reversed(widths[:-1]):
                self.upsamples.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
                self.decoders.append(ConvBlock(width * 2, width))
                previous = width
            self.output = nn.Conv2d(previous, output_channels, 1)

        def forward(self, values):
            import torch.nn.functional as functional

            height, width = values.shape[-2:]
            multiple = 2 ** max(config.depth - 1, 0)
            pad_height = (multiple - height % multiple) % multiple
            pad_width = (multiple - width % multiple) % multiple
            values = functional.pad(values, (0, pad_width, 0, pad_height))
            skips = []
            for index, encoder in enumerate(self.encoders):
                values = encoder(values)
                if index < len(self.encoders) - 1:
                    skips.append(values)
                    values = self.pool(values)
            for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
                values = upsample(values)
                values = decoder(torch.cat([values, skip], dim=1))
            return self.output(values)[..., :height, :width]

    return CompactUNet()


def train_neural_ensemble(
    signals: list[np.ndarray],
    config: AcquisitionV4NeuralConfig,
    seed: int = 0,
) -> NeuralReconstructionEnsemble:
    """Train a deterministic compact ensemble from dense training slices."""

    if not torch_available():
        raise RuntimeError("install the balance-nm learned extra to use neural reconstruction")
    if not signals:
        raise ValueError("neural ensemble training requires at least one dense training slice")
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lows, spans = _robust_channel_scale(signals)
    examples = []
    for signal in signals:
        target = _scale(signal, lows, spans)
        for _ in range(config.training_masks_per_slice):
            mask = _random_mask(signal.shape[1:], rng)
            nearest, distance = nearest_signal_and_distance(signal, mask)
            network_input = np.concatenate(
                [_scale(nearest, lows, spans), mask[None].astype(float), distance[None]],
                axis=0,
            )
            examples.append((network_input.astype(np.float32), target.astype(np.float32)))
    models = []
    for member in range(config.ensemble_size):
        torch.manual_seed(seed + member)
        model = _build_unet(signals[0].shape[0] + 2, signals[0].shape[0], config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_function = nn.HuberLoss()
        best = np.inf
        stale = 0
        for _ in range(config.epochs):
            model.train()
            losses = []
            order = rng.permutation(len(examples))
            for start in range(0, len(order), config.batch_size):
                batch = [examples[index] for index in order[start : start + config.batch_size]]
                inputs = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
                targets = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
                optimizer.zero_grad()
                loss = loss_function(model(inputs), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            current = float(np.mean(losses))
            if current < best - 1.0e-6:
                best = current
                stale = 0
            else:
                stale += 1
                if stale >= config.early_stop_patience:
                    break
        models.append(model)
    return NeuralReconstructionEnsemble(models=models, lows=lows, spans=spans, device=device)
