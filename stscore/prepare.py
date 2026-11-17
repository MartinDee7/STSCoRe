"""Prepare unnormalized training data and transferable spatial descriptors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import validate_split


def topology_descriptors(adjacency, learning_nodes):
    """Four node-ID-independent statistics, computed against fitting-side nodes.

    Adjacency must contain physical distance/edge metadata, not correlations
    estimated from held-out traffic. Training rows therefore see only the
    induced fitting graph; query rows can use their deployment-time metadata.
    """
    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("adjacency must be a finite nonnegative square metadata matrix")
    a = a.copy()
    np.fill_diagonal(a, 0)
    allowed = a[:, learning_nodes]
    connected = allowed > 0
    count = connected.sum(-1)
    mean = allowed.sum(-1) / np.maximum(count, 1)
    variance = ((allowed - mean[:, None]) ** 2 * connected).sum(-1) / np.maximum(count, 1)
    maximum = allowed.max(-1)
    return np.log1p(np.stack([count, mean, np.sqrt(variance), maximum], -1)).astype(np.float32)


def grid_descriptors(coordinates):
    """Physical grid coordinates (x/y or lon/lat); no learned sensor identities."""
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("grid coordinates must be a finite N x 2 array")
    span = np.ptp(xy, axis=0)
    normalized = 2 * (xy - xy.min(0)) / np.maximum(span, 1e-8) - 1
    return np.concatenate([normalized, normalized ** 2], -1).astype(np.float32)


def validate_milan_quadrants(coordinates, split):
    xy = np.asarray(coordinates)
    lon, lat = np.median(xy, axis=0)
    southeast = (xy[:, 0] >= lon) & (xy[:, 1] < lat)
    if southeast[split["train_nodes"]].any() or not southeast[split["test_nodes"]].all():
        raise ValueError("Milan fixed IDs do not match q1-q3 learning / southeast q4 evaluation; check cell-column order")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="unnormalized T x N [.npy] or NPZ data array")
    parser.add_argument("--input-key", default="data")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--split", required=True)
    parser.add_argument("--segments", type=int, nargs=3, required=True, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--start-step", type=int, default=0, help="explicit subset offset in already resampled raw series")
    parser.add_argument("--steps-per-day", type=int, required=True)
    parser.add_argument("--dataset", choices=["PeMSD4", "PeMSD8", "Milan"], required=True)
    metadata = parser.add_mutually_exclusive_group(required=True)
    metadata.add_argument("--adjacency", help="physical N x N adjacency/distance metadata NPY, PeMS")
    metadata.add_argument("--coordinates", help="physical N x 2 grid coordinates NPY, Milan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("output exists; use a new prepared-data path")
    if min(args.segments) < 24 or args.start_step < 0 or args.steps_per_day not in (48, 288):
        parser.error("invalid temporal dimensions or sampling frequency")
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            raw = loaded[args.input_key].copy()
        finally:
            loaded.close()
    else:
        raw = loaded
    if raw.ndim == 3:
        raw = raw[:, :, args.channel]
    if raw.ndim != 2:
        parser.error("input must have shape T x N or T x N x channels")
    stop = args.start_step + sum(args.segments)
    if stop > len(raw):
        parser.error("declared time segments exceed the available raw series")
    split = json.loads(Path(args.split).read_text())
    validate_split(split, raw.shape[1])
    if args.dataset == "Milan":
        if not args.coordinates:
            parser.error("Milan requires coordinates to enforce the geographical split")
        xy = np.load(args.coordinates, allow_pickle=False)
        if len(xy) != raw.shape[1]:
            parser.error("coordinate/data column mismatch")
        validate_milan_quadrants(xy, split)
        spatial = grid_descriptors(xy)
    else:
        if not args.adjacency:
            parser.error("PeMS requires physical adjacency metadata")
        spatial = topology_descriptors(np.load(args.adjacency, allow_pickle=False), split["train_nodes"])
    if len(spatial) != raw.shape[1]:
        parser.error("spatial metadata/data column mismatch")
    series = np.asarray(raw[args.start_step:stop], dtype=np.float32)
    ntrain, nval, _ = args.segments
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, schema_version=np.array(1), normalized=np.array(False),
                        dataset_name=np.array(args.dataset), steps_per_day=np.array(args.steps_per_day),
                        time_start=np.array(args.start_step), spatial=spatial,
                        train=series[:ntrain], val=series[ntrain:ntrain+nval], test=series[ntrain+nval:])
    print(f"Prepared {args.dataset}: segments={args.segments}, sensors={raw.shape[1]}; scaler will fit only permitted training windows.")


if __name__ == "__main__":
    main()
