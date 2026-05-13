from __future__ import annotations

from perovskite_screening.io.run_artifacts import (
    cache_dir_rel_path,
    history_dir_rel_path,
    history_file_rel_path,
    model_dir_rel_path,
    model_file_rel_path,
    prediction_rel_path,
    run_artifact_stem,
)


def test_run_artifact_paths_use_one_stem_convention() -> None:
    stem = run_artifact_stem(
        split_strategy="element_set",
        model_name="matgl_megnet_frozen",
        budget_name="B500",
        seed=42,
    )

    assert stem == "element_set_matgl_megnet_frozen_B500_seed42"
    assert prediction_rel_path(stem) == "outputs/runs/predictions/element_set_matgl_megnet_frozen_B500_seed42.csv"
    assert history_file_rel_path(stem) == "outputs/runs/history/element_set_matgl_megnet_frozen_B500_seed42.csv"
    assert history_dir_rel_path(stem) == "outputs/runs/history/element_set_matgl_megnet_frozen_B500_seed42"
    assert (
        model_file_rel_path(model_family="cgcnn", stem=stem, suffix=".pt")
        == "outputs/models/cgcnn/element_set_matgl_megnet_frozen_B500_seed42.pt"
    )
    assert (
        model_dir_rel_path(model_family="matgl", stem=stem)
        == "outputs/models/matgl/element_set_matgl_megnet_frozen_B500_seed42"
    )
    assert (
        cache_dir_rel_path(cache_family="matgl", stem=stem)
        == "outputs/cache/matgl/element_set_matgl_megnet_frozen_B500_seed42"
    )
    assert (
        cache_dir_rel_path(cache_family="matgl", stem=stem, partition="train")
        == "outputs/cache/matgl/element_set_matgl_megnet_frozen_B500_seed42/train"
    )
