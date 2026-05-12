from __future__ import annotations

from perovskite_screening.models.descriptor import build_descriptor_model


MODEL_BUILDERS = {
    "descriptor": build_descriptor_model,
}
