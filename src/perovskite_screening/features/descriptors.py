from __future__ import annotations

import numpy as np
import pandas as pd


ALLOWED_FEATURE_SETS = ("starter", "expanded")


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _weighted_stats(values: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_range": float("nan"),
        }
    values = values[valid]
    weights = weights[valid] / weights[valid].sum()
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": float(np.sqrt(max(var, 0.0))),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def _composition_entropy(amounts: np.ndarray) -> float:
    amounts = np.asarray(amounts, dtype=float)
    amounts = amounts[amounts > 0]
    if len(amounts) == 0:
        return float("nan")
    p = amounts / amounts.sum()
    return float(-np.sum(p * np.log(p)))


def starter_features(structure) -> dict[str, float]:
    composition = structure.composition
    atomic_numbers = np.array([element.Z for element in composition.elements], dtype=float)
    return {
        "n_sites": float(structure.num_sites),
        "volume": float(structure.volume),
        "density": float(structure.density),
        "num_unique_elements": float(len(composition.elements)),
        "mean_atomic_number": float(atomic_numbers.mean()),
        "std_atomic_number": float(atomic_numbers.std()),
    }


def expanded_features(structure) -> dict[str, float]:
    composition = structure.composition
    elements = list(composition.elements)
    amounts = np.array([composition[element] for element in elements], dtype=float)
    features = dict(starter_features(structure))
    lattice = structure.lattice
    fractions = amounts / amounts.sum()
    features.update(
        {
            "volume_per_site": float(structure.volume / structure.num_sites),
            "lattice_a": float(lattice.a),
            "lattice_b": float(lattice.b),
            "lattice_c": float(lattice.c),
            "lattice_alpha": float(lattice.alpha),
            "lattice_beta": float(lattice.beta),
            "lattice_gamma": float(lattice.gamma),
            "composition_entropy": _composition_entropy(amounts),
            "max_element_fraction": float(fractions.max()),
            "min_element_fraction": float(fractions.min()),
        }
    )
    elemental_properties = {
        "atomic_number": [element.Z for element in elements],
        "atomic_mass": [_as_float(element.atomic_mass) for element in elements],
        "electronegativity": [_as_float(element.X) for element in elements],
        "atomic_radius": [_as_float(element.atomic_radius) for element in elements],
        "mendeleev_no": [_as_float(element.mendeleev_no) for element in elements],
        "periodic_row": [_as_float(element.row) for element in elements],
        "periodic_group": [_as_float(element.group) for element in elements],
    }
    for prop_name, values in elemental_properties.items():
        features.update(_weighted_stats(np.array(values, dtype=float), amounts, prop_name))
    return features


def featurize(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "starter":
        rows = [starter_features(structure) for structure in df["structure"]]
    elif feature_set == "expanded":
        rows = [expanded_features(structure) for structure in df["structure"]]
    else:
        raise ValueError(f"Unknown feature_set={feature_set!r}")
    return pd.DataFrame(rows, index=df.index)
