from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def build_descriptor_model(model_kind: str, seed: int):
    if model_kind == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            random_state=int(seed),
            n_jobs=-1,
            min_samples_leaf=1,
        )
    elif model_kind == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install it with `pip install -e '.[xgb]'` "
                "or `pip install -r requirements/xgb.txt`."
            ) from exc
        regressor = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        raise ValueError(f"Unknown descriptor model kind: {model_kind!r}")
    return make_pipeline(SimpleImputer(strategy="median"), regressor)
