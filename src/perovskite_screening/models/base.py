from __future__ import annotations

from typing import Protocol


class RegressionModel(Protocol):
    def fit(self, x, y) -> object:
        ...

    def predict(self, x):
        ...
