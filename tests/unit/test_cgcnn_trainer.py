from __future__ import annotations

from perovskite_screening.config import ProjectConfig
from perovskite_screening.training.cgcnn_trainer import cgcnn_model_params, update_early_stopping_state


def test_cgcnn_config_params_are_parsed() -> None:
    params = cgcnn_model_params(ProjectConfig.from_file("configs/experiments/cgcnn.yaml"))

    assert params["epochs"] == 100
    assert params["batch_size"] == 128
    assert params["lr"] == 0.0003
    assert params["weight_decay"] == 0.00001
    assert params["scheduler_patience"] == 8
    assert params["early_stopping_patience"] == 15
    assert params["monitor"] == "val_mae"


def test_cgcnn_early_stopping_state_tracks_best_and_patience() -> None:
    improved, best, bad_epochs, should_stop = update_early_stopping_state(
        val_mae=0.5,
        best_val_mae=float("inf"),
        bad_epochs=0,
        patience=2,
    )
    assert improved is True
    assert best == 0.5
    assert bad_epochs == 0
    assert should_stop is False

    improved, best, bad_epochs, should_stop = update_early_stopping_state(
        val_mae=0.6,
        best_val_mae=best,
        bad_epochs=bad_epochs,
        patience=2,
    )
    assert improved is False
    assert best == 0.5
    assert bad_epochs == 1
    assert should_stop is False

    improved, best, bad_epochs, should_stop = update_early_stopping_state(
        val_mae=0.7,
        best_val_mae=best,
        bad_epochs=bad_epochs,
        patience=2,
    )
    assert improved is False
    assert best == 0.5
    assert bad_epochs == 2
    assert should_stop is True


def test_cgcnn_early_stopping_can_be_disabled() -> None:
    _, _, _, should_stop = update_early_stopping_state(
        val_mae=1.0,
        best_val_mae=0.5,
        bad_epochs=100,
        patience=None,
    )
    assert should_stop is False
