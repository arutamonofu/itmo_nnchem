from __future__ import annotations

import builtins

import pytest

from perovskite_screening.models.descriptor import build_descriptor_model


def test_xgb_missing_dependency_message_mentions_project_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("missing xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"\.\[xgb\]"):
        build_descriptor_model("xgb", seed=42)
