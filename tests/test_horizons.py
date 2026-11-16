"""Tests for the forecast-horizon experiment. Author: plholt3"""

import numpy as np
import pandas as pd

from energy_forecasting.data import TARGET_COLUMN
from energy_forecasting.features import build_features
from energy_forecasting.horizon_experiment import (
    TREE_DEPTHS,
    TREE_MIN_SAMPLES_LEAF,
    component_tables,
    horizon_candidates,
    nnmf_variance_share,
    pca_variance_share,
    purged_split,
    rank_agreement,
    shift_target,
    side_by_side,
    target_alignment,
    tree_depth_curve,
)
from energy_forecasting.modeling import tune_models_time_series


def hourly_frame(periods: int = 600) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = 30_000 + 1_500 * np.sin(np.arange(periods) * 2 * np.pi / 24) + np.arange(periods)
    return pd.DataFrame({TARGET_COLUMN: values}, index=index)


def test_horizon_one_leaves_target_unchanged():
    frame = build_features(hourly_frame())
    pd.testing.assert_frame_equal(shift_target(frame, 1), frame)


def test_shifted_target_is_load_horizon_minus_one_hours_later():
    raw = hourly_frame()
    frame = build_features(raw)
    shifted = shift_target(frame, 24)
    row = shifted.index[10]
    assert shifted.loc[row, TARGET_COLUMN] == raw.loc[row + pd.Timedelta(hours=23), TARGET_COLUMN]
    # Features are untouched and the last 23 rows have no future target.
    assert shifted.loc[row, "lag_1"] == frame.loc[row, "lag_1"]
    assert len(shifted) == len(frame) - 23


def test_shift_uses_timestamps_not_row_positions():
    frame = build_features(hourly_frame())
    gapped = frame.drop(frame.index[50])
    shifted = shift_target(gapped, 3)
    row = frame.index[48]  # its target hour (row 50) was removed
    assert row not in shifted.index
    assert shifted.loc[frame.index[47], TARGET_COLUMN] == frame.loc[frame.index[49], TARGET_COLUMN]


def test_purged_split_keeps_every_training_target_before_the_test_period():
    frame = shift_target(build_features(hourly_frame()), 24)
    split_time = frame.index[300]
    train, test = purged_split(frame, split_time, 24)
    last_target_time = train.index.max() + pd.Timedelta(hours=23)
    assert last_target_time < split_time
    assert test.index.min() == split_time


def test_tuning_gap_separates_fit_and_validation_blocks():
    frame = build_features(hourly_frame())
    candidates = {"decision_tree": [{"max_depth": 3, "min_samples_leaf": 2}]}
    _, scores = tune_models_time_series(frame, n_splits=2, candidates=candidates, gap=23)
    for row in scores.itertuples():
        validation_position = frame.index.get_loc(row.validation_start)
        assert row.train_rows + 23 == validation_position


def test_variance_rankings_score_every_feature_between_zero_and_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    features = pd.DataFrame(
        {
            "a": base + 5,
            "b": 2 * base + rng.normal(scale=0.05, size=400) + 10,
            "noise": rng.normal(size=400) + 5,
        }
    )
    pca, _ = pca_variance_share(features)
    nnmf, _ = nnmf_variance_share(features)
    for scores in (pca, nnmf):
        assert list(scores.index) == ["a", "b", "noise"]
        assert ((scores >= 0) & (scores <= 1)).all()
    # Two correlated features dominate the first component.
    assert pca["a"] > 0.9 and pca["b"] > 0.9


def test_target_alignment_ranks_the_feature_that_tracks_the_target_first():
    rng = np.random.default_rng(1)
    target = pd.Series(rng.normal(size=400))
    features = pd.DataFrame(
        {"tracks": target + rng.normal(scale=0.1, size=400), "unrelated": rng.normal(size=400)}
    )
    alignment, captured = target_alignment(features, target)
    assert alignment["tracks"] > alignment["unrelated"]
    assert 0 <= captured <= 1


def test_rank_agreement_reports_each_model_horizon_and_reference():
    features = ["a", "b", "c"]
    importance = pd.DataFrame(
        {
            "feature": features * 2,
            "impurity": [0.6, 0.3, 0.1, 0.5, 0.4, 0.1],
            "permutation_mae_increase": [30.0, 10.0, 1.0, 20.0, 15.0, 2.0],
            "horizon": ["1 hour"] * 3 + ["1 day"] * 3,
            "model": ["random_forest"] * 6,
        }
    )
    variance = pd.DataFrame({"pca": [0.9, 0.5, 0.1], "nnmf": [0.8, 0.6, 0.2]}, index=features)
    alignment = pd.DataFrame(
        {"feature": features * 2, "target_pca": [0.9, 0.4, 0.1] * 2,
         "horizon": ["1 hour"] * 3 + ["1 day"] * 3}
    )
    agreement = rank_agreement(importance, variance, alignment)
    assert len(agreement) == 2 * 2 * 3
    assert np.isclose(agreement["spearman"], 1.0).all()


def test_horizon_grid_searches_shallow_trees_and_keeps_forest_grid():
    candidates = horizon_candidates()
    depths = {item["max_depth"] for item in candidates["decision_tree"]}
    assert min(depths) == min(TREE_DEPTHS) and set(TREE_DEPTHS) == depths
    assert len(candidates["decision_tree"]) == len(TREE_DEPTHS) * len(TREE_MIN_SAMPLES_LEAF)
    assert len(candidates["random_forest"]) == 4


def test_component_tables_have_one_row_per_feature():
    rng = np.random.default_rng(2)
    features = pd.DataFrame(rng.random((200, 4)), columns=["a", "b", "c", "d"])
    pca, nnmf = component_tables(features, pca_components=2, nnmf_rank=3)
    assert list(pca.index) == list(features.columns) and list(pca.columns) == ["PC1", "PC2"]
    assert list(nnmf.columns) == ["Factor1", "Factor2", "Factor3"]
    assert np.allclose(nnmf.sum(axis=0), 1.0)


def test_depth_curve_training_error_never_rises_with_depth():
    frame = build_features(hourly_frame())
    train, test = frame.iloc[:400], frame.iloc[400:]
    columns = [column for column in frame.columns if column != TARGET_COLUMN]
    curve = tree_depth_curve(train, test, columns, min_samples_leaf=2)
    assert (np.diff(curve["train_mae_mw"].to_numpy()) <= 1e-9).all()
    assert np.allclose(curve["gap_mw"], curve["holdout_mae_mw"] - curve["train_mae_mw"])


def test_side_by_side_orders_each_column_by_its_own_ranking():
    importance = pd.DataFrame(
        {
            "feature": ["a", "b"] * 2,
            "permutation_mae_increase": [1.0, 5.0, 4.0, 2.0],
            "horizon": ["1 hour"] * 4,
            "model": ["decision_tree"] * 2 + ["random_forest"] * 2,
        }
    )
    variance = pd.DataFrame({"pca": [0.9, 0.1], "nnmf": [0.2, 0.8]}, index=["a", "b"])
    table = side_by_side(importance, variance, ["1 hour"])
    assert table.loc[1, "PCA variance captured"].startswith("a")
    assert table.loc[1, "NNMF variance captured"].startswith("b")
    assert table.loc[1, "Decision tree 1 hour"].startswith("b")
    assert table.loc[1, "Random forest 1 hour"].startswith("a")
