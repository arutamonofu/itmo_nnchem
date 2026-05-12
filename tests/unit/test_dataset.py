from __future__ import annotations

import pandas as pd

from perovskite_screening.data import dataset


def test_load_matbench_perovskites_uses_local_raw_cache(tmp_path, monkeypatch) -> None:
    raw_path = tmp_path / "matbench_perovskites.pkl"
    pd.DataFrame(
        {
            "structure": ["s0", "s1"],
            "formation_energy": [-1.0, -2.0],
        }
    ).to_pickle(raw_path)
    monkeypatch.setattr(dataset, "raw_dataset_path", lambda dataset_name: raw_path)

    df = dataset.load_matbench_perovskites("matbench_perovskites")

    assert list(df.columns) == ["sample_id", "structure", "target"]
    assert df["sample_id"].tolist() == [0, 1]
    assert df["target"].tolist() == [-1.0, -2.0]
