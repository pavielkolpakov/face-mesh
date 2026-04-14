# mapping

Measurements → blendshape params (0..1). Per-param routing: regression when trained, else heuristic.

- `estimator.py` — `Estimator` protocol: `is_trained`, `feature_names`, `fit(X, y)`, `predict(x) -> float`.
- `heuristic.py` — `HeuristicEstimator(feature_name, min, max)` — single-feature min-max + clamp. Never trained.
- `regression.py` — (todo, Phase 4) Ridge wrapper; features = full measurement vector + `frontal_score`.
- `mapper.py` — `FaceMapper` / `BodyMapper`. Dict of `{param_name: Estimator}`. `from_heuristic_bounds(bounds)` builds the defaults; `save/load` uses joblib.

Rules:
- Heuristic bounds live in `config/heuristic_bounds.yaml` — don't hardcode new ones in Python.
- Mapper artifacts: `models/face_mapper.joblib`, `models/body_mapper.joblib`.
