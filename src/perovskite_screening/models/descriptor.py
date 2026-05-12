from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


RF_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "n_jobs": -1,
    "min_samples_leaf": 1,
}

XGB_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "tree_method": "hist",
}
XGB_ALLOWED_EXTRA_PARAMS = {
    "gamma",
    "grow_policy",
    "max_bin",
    "min_child_weight",
    "reg_alpha",
}


def _validate_params(model_kind: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    unsupported = sorted(set(params) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported {model_kind} model.params keys: {unsupported}")


def _rf_allowed_params() -> set[str]:
    return set(RandomForestRegressor().get_params())


def _xgb_allowed_params(regressor_class) -> set[str]:
    try:
        return set(regressor_class().get_params()) | XGB_ALLOWED_EXTRA_PARAMS
    except Exception:
        return set(XGB_DEFAULT_PARAMS) | XGB_ALLOWED_EXTRA_PARAMS


def _merged_params(
    defaults: Mapping[str, Any],
    params: Mapping[str, Any] | None,
    seed: int,
    *,
    model_kind: str,
    allowed: set[str],
) -> dict[str, Any]:
    merged = dict(defaults)
    if params:
        _validate_params(model_kind, params, allowed | {"random_state"})
        merged.update(dict(params))
    merged["random_state"] = int(seed)
    return merged


def descriptor_effective_params(model_kind: str, seed: int, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if model_kind == "rf":
        return _merged_params(
            RF_DEFAULT_PARAMS,
            params,
            seed,
            model_kind=model_kind,
            allowed=_rf_allowed_params(),
        )
    if model_kind == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it with `pip install -e '.[xgb]'` "
                "or `pip install -r requirements/xgb.txt`."
            ) from exc
        return _merged_params(
            XGB_DEFAULT_PARAMS,
            params,
            seed,
            model_kind=model_kind,
            allowed=_xgb_allowed_params(XGBRegressor),
        )
    raise ValueError(f"Unknown descriptor model kind: {model_kind!r}")


def build_descriptor_model(model_kind: str, seed: int, params: Mapping[str, Any] | None = None):
    if model_kind == "rf":
        regressor = RandomForestRegressor(**descriptor_effective_params(model_kind, seed, params))
    elif model_kind == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it with `pip install -e '.[xgb]'` "
                "or `pip install -r requirements/xgb.txt`."
            ) from exc
        regressor = XGBRegressor(**descriptor_effective_params(model_kind, seed, params))
    else:
        raise ValueError(f"Unknown descriptor model kind: {model_kind!r}")
    return make_pipeline(SimpleImputer(strategy="median"), regressor)
