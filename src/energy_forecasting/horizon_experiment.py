"""Compare tree-model feature importance across forecast horizons with PCA and NNMF.

The feature pipeline is unchanged. Each row keeps the features known at its forecast time
(data through hour t - 1); only the target moves. Row t predicts load at t + horizon - 1, so
horizon 1 reproduces the existing one-hour-ahead setup, 24 is one day ahead, and 168 is one
week ahead.

The variance side runs PCA and NNMF on the same feature table the models use, so every
feature gets a variance score that lines up one-to-one with its model importance.

Author: plholt3
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import NMF, PCA
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from energy_forecasting.data import TARGET_COLUMN
from energy_forecasting.evaluation import chronological_split, regression_metrics
from energy_forecasting.features import feature_columns
from energy_forecasting.final_experiment import aligned_feature_sets
from energy_forecasting.modeling import (
    evaluate_models,
    tune_models_time_series,
    tuning_candidates,
)

HORIZONS = {"1 hour": 1, "1 day": 24, "1 week": 168}
TREE_MODELS = ("decision_tree", "random_forest")
VARIANCE_TARGET = 0.95

# Wider decision-tree search, including shallow trees, so cross-validation can find the depth
# where holdout error stops improving at each horizon. The random forest keeps the team grid.
TREE_DEPTHS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16)
TREE_MIN_SAMPLES_LEAF = (2, 8, 32)
DEPTH_CURVE_DEPTHS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, None)


def horizon_candidates() -> dict[str, list[dict]]:
    """Tuning grid: wide decision-tree depth search plus the team's random-forest grid."""
    return {
        "decision_tree": [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in TREE_DEPTHS
            for leaf in TREE_MIN_SAMPLES_LEAF
        ],
        "random_forest": tuning_candidates()["random_forest"],
    }
RANDOM_STATE = 0

# Chart colours: one categorical slot per horizon, light surface and text tokens.
HORIZON_COLORS = {"1 hour": "#2a78d6", "1 day": "#eb6834", "1 week": "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


# --------------------------------------------------------------------------------------
# Target shifting and leakage-safe splitting
# --------------------------------------------------------------------------------------


def shift_target(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Replace the target with load ``horizon - 1`` hours later; features stay as they are.

    The shift uses timestamps, not row positions, so a missing hour can never misalign a
    target. Rows whose future target falls past the end of the data are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    shifted = frame.copy()
    future = frame[TARGET_COLUMN].shift(-(horizon - 1), freq="h")
    shifted[TARGET_COLUMN] = future.reindex(frame.index)
    return shifted.dropna(subset=[TARGET_COLUMN])


def purged_split(
    frame: pd.DataFrame, split_time: pd.Timestamp, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at ``split_time`` and drop training rows whose target lands in the test period."""
    cutoff = split_time - pd.Timedelta(hours=horizon - 1)
    train = frame[frame.index < cutoff].copy()
    test = frame[frame.index >= split_time].copy()
    return train, test


# --------------------------------------------------------------------------------------
# Variance rankings (PCA and NNMF on the model's feature table)
# --------------------------------------------------------------------------------------


def _components_for(cumulative: np.ndarray, target: float = VARIANCE_TARGET) -> int:
    return int(min(np.searchsorted(cumulative, target) + 1, len(cumulative)))


def pca_variance_share(features: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    """Share of each standardized feature's variance captured by the PCs kept for 95%.

    This is each feature's communality: the sum over kept components of its squared loading
    weighted by that component's variance.
    """
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(random_state=RANDOM_STATE).fit(standardized)
    k = _components_for(np.cumsum(pca.explained_variance_ratio_))
    loadings = pca.components_[:k].T * np.sqrt(pca.explained_variance_[:k])
    captured = np.clip((loadings**2).sum(axis=1), 0, 1)
    info = {
        "components_kept": k,
        "variance_explained": float(pca.explained_variance_ratio_[:k].sum()),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
    }
    return pd.Series(captured, index=features.columns, name="pca"), info


def _reconstruction_quality(actual: np.ndarray, reconstruction: np.ndarray) -> float:
    residual = np.square(actual - reconstruction).sum()
    baseline = np.square(actual - actual.mean()).sum()
    return float(1 - residual / baseline) if baseline else 1.0


def nnmf_variance_share(
    features: pd.DataFrame, max_iter: int = 1000
) -> tuple[pd.Series, dict[str, float]]:
    """Share of each scaled feature's variance captured by the NNMF rank that reaches 95%.

    Columns are min-max scaled to 0-1 (NMF needs non-negative input). The rank is chosen the
    same way as the team's NNMF script: the smallest rank whose reconstruction quality reaches
    95%. Each feature's score is the R-squared of its own reconstructed column.
    """
    scaled = MinMaxScaler().fit_transform(features)
    reconstruction = scaled
    chosen_rank, quality = features.shape[1], 1.0
    for rank in range(1, features.shape[1] + 1):
        model = NMF(
            n_components=rank, init="nndsvda", random_state=RANDOM_STATE, max_iter=max_iter
        )
        scores = model.fit_transform(scaled)
        reconstruction = scores @ model.components_
        quality = _reconstruction_quality(scaled, reconstruction)
        if quality >= VARIANCE_TARGET:
            chosen_rank = rank
            break
    residual = np.square(scaled - reconstruction).sum(axis=0)
    total = np.square(scaled - scaled.mean(axis=0)).sum(axis=0)
    captured = np.clip(1 - np.divide(residual, total, out=np.ones_like(total), where=total > 0), 0, 1)
    info = {"rank": chosen_rank, "reconstruction_quality": quality}
    return pd.Series(captured, index=features.columns, name="nnmf"), info


def variance_curves(
    features: pd.DataFrame, max_iter: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full PCA cumulative-variance curve and NNMF reconstruction quality at every rank."""
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(random_state=RANDOM_STATE).fit(standardized)
    pca_curve = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    scaled = MinMaxScaler().fit_transform(features)
    rows = []
    for rank in range(1, features.shape[1] + 1):
        model = NMF(
            n_components=rank, init="nndsvda", random_state=RANDOM_STATE, max_iter=max_iter
        )
        scores = model.fit_transform(scaled)
        quality = _reconstruction_quality(scaled, scores @ model.components_)
        rows.append({"rank": rank, "reconstruction_quality": quality,
                     "iterations": model.n_iter_})
    return pca_curve, pd.DataFrame(rows)


def write_variance_curves(features: pd.DataFrame, output: Path) -> None:
    """Save the PCA and NNMF curves for the model feature table as CSVs and charts."""
    output.mkdir(parents=True, exist_ok=True)
    pca_curve, nnmf_curve = variance_curves(features)
    pca_curve.to_csv(output / "pca_cumulative_variance.csv", index=False)
    nnmf_curve.to_csv(output / "nnmf_rank_quality.csv", index=False)
    plot_variance_curves(pca_curve, nnmf_curve, output)


def plot_variance_curves(pca_curve: pd.DataFrame, nnmf_curve: pd.DataFrame, output: Path) -> None:
    """Draw both variance charts from their CSV tables (so they can be redrawn without refitting)."""
    _plot_curve(
        pca_curve["component"], pca_curve["cumulative_variance"],
        "Principal component (PC1 = largest; cumulative line includes every PC up to this one)",
        "Variance explained (% of total)",
        "PCA on the model feature table: cumulative variance explained",
        output / "pca_cumulative_variance.png",
        bars=pca_curve["explained_variance_ratio"],
        tick_prefix="PC",
    )
    _plot_curve(
        nnmf_curve["rank"], nnmf_curve["reconstruction_quality"],
        "NNMF rank (number of factors)",
        "Reconstruction quality (R², % of variance rebuilt)",
        "NNMF on the model feature table: reconstruction quality by rank",
        output / "nnmf_rank_quality.png",
    )


def component_tables(
    features: pd.DataFrame, pca_components: int, nnmf_rank: int, max_iter: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What each kept PCA component and NNMF factor is made of, one row per feature.

    PCA values are loadings (correlation between the standardized feature and the component).
    NNMF values are each factor's weights normalized to sum to 1, i.e. the share of the factor
    contributed by each min-max scaled feature.
    """
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(random_state=RANDOM_STATE).fit(standardized)
    loadings = pca.components_[:pca_components].T * np.sqrt(pca.explained_variance_[:pca_components])
    pca_table = pd.DataFrame(
        loadings,
        index=features.columns,
        columns=[f"PC{i + 1}" for i in range(pca_components)],
    )
    scaled = MinMaxScaler().fit_transform(features)
    model = NMF(
        n_components=nnmf_rank, init="nndsvda", random_state=RANDOM_STATE, max_iter=max_iter
    )
    model.fit(scaled)
    weights = model.components_ / model.components_.sum(axis=1, keepdims=True)
    nnmf_table = pd.DataFrame(
        weights.T,
        index=features.columns,
        columns=[f"Factor{i + 1}" for i in range(nnmf_rank)],
    )
    return pca_table.rename_axis("feature"), nnmf_table.rename_axis("feature")


def tree_depth_curve(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], min_samples_leaf: int
) -> pd.DataFrame:
    """Training vs holdout error of a decision tree at each max depth (overfitting check)."""
    rows = []
    for depth in DEPTH_CURVE_DEPTHS:
        model = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=42
        )
        model.fit(train[columns], train[TARGET_COLUMN])
        train_fit = regression_metrics(train[TARGET_COLUMN], model.predict(train[columns]))
        holdout_fit = regression_metrics(test[TARGET_COLUMN], model.predict(test[columns]))
        train_mae, holdout_mae = train_fit["mae"], holdout_fit["mae"]
        rows.append(
            {
                "max_depth": "none" if depth is None else depth,
                "min_samples_leaf": min_samples_leaf,
                "train_mae_mw": train_mae,
                "holdout_mae_mw": holdout_mae,
                "gap_mw": holdout_mae - train_mae,
                "train_mape_percent": train_fit["mape_percent"],
                "holdout_mape_percent": holdout_fit["mape_percent"],
                "number_of_leaves": int(model.get_n_leaves()),
                "avg_train_rows_per_leaf": len(train) / model.get_n_leaves(),
            }
        )
    return pd.DataFrame(rows)


def target_alignment(features: pd.DataFrame, target: pd.Series) -> tuple[pd.Series, float]:
    """PCA on features plus the shifted target: how strongly each feature moves with the target.

    Returns the absolute correlation between each feature and the target as rebuilt from the
    components kept for 95% of variance, and how much of the target's own variance those
    components capture.
    """
    table = features.assign(__target__=target.to_numpy())
    standardized = StandardScaler().fit_transform(table)
    pca = PCA(random_state=RANDOM_STATE).fit(standardized)
    k = _components_for(np.cumsum(pca.explained_variance_ratio_))
    loadings = pca.components_[:k].T * np.sqrt(pca.explained_variance_[:k])
    target_loading = loadings[-1]
    alignment = np.abs(loadings[:-1] @ target_loading)
    target_captured = float(np.clip((target_loading**2).sum(), 0, 1))
    return pd.Series(alignment, index=features.columns, name="target_pca"), target_captured


# --------------------------------------------------------------------------------------
# Model importances
# --------------------------------------------------------------------------------------


def model_importances(
    model, test: pd.DataFrame, columns: list[str], repeats: int
) -> pd.DataFrame:
    """Built-in (impurity) importance and permutation importance on the holdout."""
    permutation = permutation_importance(
        model,
        test[columns],
        test[TARGET_COLUMN],
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame(
        {
            "feature": columns,
            "impurity": model.feature_importances_,
            "permutation_mae_increase": permutation.importances_mean,
            "permutation_std": permutation.importances_std,
        }
    )


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


def run(
    data_path: Path,
    weather_path: Path,
    output: Path,
    horizons: dict[str, int] = HORIZONS,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
    permutation_repeats: int = 5,
) -> dict[str, pd.DataFrame]:
    frames = aligned_feature_sets(data_path, weather_path)
    # One split time for every horizon, so all horizons forecast from the same origins.
    _, base_test = chronological_split(frames["weather_enabled"], test_fraction)
    split_time = base_test.index.min()

    metrics_rows, cv_tables, importance_tables, alignment_tables = [], [], [], []
    depth_tables, final_rows = [], []
    target_capture: dict[str, float] = {}
    variance = None

    for label, horizon in horizons.items():
        print(f"\n=== {label} ahead (target shifted {horizon - 1} h) ===", flush=True)
        splits = {
            name: purged_split(shift_target(frame, horizon), split_time, horizon)
            for name, frame in frames.items()
        }
        weather_train = splits["weather_enabled"][0]
        columns = feature_columns(weather_train)

        if variance is None:
            # Features are identical at every horizon, so the variance ranking is computed once
            # on the one-hour training period.
            pca_share, pca_info = pca_variance_share(weather_train[columns])
            nnmf_share, nnmf_info = nnmf_variance_share(weather_train[columns])
            variance = pd.concat([pca_share, nnmf_share], axis=1)
            variance_info = {"pca": pca_info, "nnmf": nnmf_info}
            print(f"PCA: {pca_info} | NNMF: {nnmf_info}", flush=True)
            write_variance_curves(weather_train[columns], output)
            pca_table, nnmf_table = component_tables(
                weather_train[columns], pca_info["components_kept"], nnmf_info["rank"]
            )
            pca_table.to_csv(output / "pca_components.csv")
            nnmf_table.to_csv(output / "nnmf_components.csv")

        alignment, captured = target_alignment(
            weather_train[columns], weather_train[TARGET_COLUMN]
        )
        alignment_tables.append(alignment.rename_axis("feature").reset_index().assign(horizon=label))
        target_capture[label] = captured

        selected, cv_results = tune_models_time_series(
            weather_train, n_splits=cv_splits, gap=horizon - 1,
            candidates=horizon_candidates(),
        )
        cv_tables.append(cv_results.assign(horizon=label))
        print(f"Selected parameters: {selected}", flush=True)

        weather_test = splits["weather_enabled"][1]
        depth_tables.append(
            tree_depth_curve(
                weather_train, weather_test, columns,
                selected["decision_tree"]["min_samples_leaf"],
            ).assign(
                horizon=label,
                selected_depth=str(selected["decision_tree"]["max_depth"] or "none"),
            )
        )

        for feature_set, (train, test) in splits.items():
            for result in evaluate_models(train, test, model_parameters=selected):
                metrics_rows.append(
                    {"horizon": label, "feature_set": feature_set, "model": result.name,
                     **result.metrics}
                )
                if feature_set != "weather_enabled":
                    continue
                final_rows.append(_final_row(label, result, selected))
                if result.name in TREE_MODELS:
                    table = model_importances(
                        result.model, test, columns, permutation_repeats
                    )
                    importance_tables.append(table.assign(horizon=label, model=result.name))
                    print(f"{result.name}: MAE {result.metrics['mae']:.1f} MW", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    importance = pd.concat(importance_tables, ignore_index=True)
    alignment = pd.concat(alignment_tables, ignore_index=True)
    agreement = rank_agreement(importance, variance, alignment)

    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(cv_tables, ignore_index=True).to_csv(output / "cv_results.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    variance.rename_axis("feature").to_csv(output / "variance_ranking.csv")
    alignment.to_csv(output / "target_alignment.csv", index=False)
    agreement.to_csv(output / "rank_agreement.csv", index=False)
    (output / "variance_settings.json").write_text(
        json.dumps({**variance_info, "target_variance_captured": target_capture}, indent=2) + "\n"
    )
    depth_curve = pd.concat(depth_tables, ignore_index=True)
    depth_curve.to_csv(output / "tree_depth_curve.csv", index=False)
    pd.DataFrame(final_rows).to_csv(output / "final_model_errors.csv", index=False)

    labels = list(horizons)
    side_by_side(importance, variance, labels).to_csv(
        output / "feature_rank_side_by_side.csv", index_label="rank"
    )
    _plot_rank_heatmap(importance, variance, labels, output)
    _plot_agreement(agreement, labels, output)
    _plot_mae(metrics, labels, output)
    _plot_depth_curve(depth_curve, labels, output)
    _write_summary(output, metrics, importance, variance, variance_info, agreement, labels)
    return {"metrics": metrics, "importance": importance, "agreement": agreement}


def _final_row(label: str, result, selected: dict) -> dict:
    """One row of the final error table (weather-enabled models at their tuned settings)."""
    row = {
        "horizon": label,
        "model": result.name,
        "mae_mw": result.metrics["mae"],
        "rmse_mw": result.metrics["rmse"],
        "mape_percent": result.metrics["mape_percent"],
        "max_depth": None,
        "min_samples_leaf": None,
        "number_of_leaves": None,
    }
    if result.name in selected:
        row["max_depth"] = selected[result.name].get("max_depth") or "none"
        row["min_samples_leaf"] = selected[result.name].get("min_samples_leaf")
        if result.name == "decision_tree":
            row["number_of_leaves"] = int(result.model.get_n_leaves())
        else:
            row["number_of_leaves"] = float(
                np.mean([tree.get_n_leaves() for tree in result.model.estimators_])
            )
    return row


def side_by_side(
    importance: pd.DataFrame, variance: pd.DataFrame, labels: list[str]
) -> pd.DataFrame:
    """Features ordered by each ranking, one column per ranking (row 1 = top feature)."""
    columns = {
        "PCA variance captured": variance["pca"].sort_values(ascending=False),
        "NNMF variance captured": variance["nnmf"].sort_values(ascending=False),
    }
    for model in TREE_MODELS:
        name = "Decision tree" if model == "decision_tree" else "Random forest"
        for label in labels:
            values = importance[
                (importance["model"] == model) & (importance["horizon"] == label)
            ].set_index("feature")["permutation_mae_increase"]
            columns[f"{name} {label}"] = values.sort_values(ascending=False)
    table = pd.DataFrame(
        {
            name: [f"{feature} ({value:.3g})" for feature, value in ranking.items()]
            for name, ranking in columns.items()
        }
    )
    table.index = range(1, len(table) + 1)
    return table


def rank_agreement(
    importance: pd.DataFrame, variance: pd.DataFrame, alignment: pd.DataFrame
) -> pd.DataFrame:
    """Spearman rank correlation between each model ranking and each variance ranking."""
    rows = []
    for (horizon, model), group in importance.groupby(["horizon", "model"], sort=False):
        group = group.set_index("feature")
        aligned = alignment[alignment["horizon"] == horizon].set_index("feature")["target_pca"]
        references = {"pca": variance["pca"], "nnmf": variance["nnmf"], "target_pca": aligned}
        for measure in ("impurity", "permutation_mae_increase"):
            for reference, values in references.items():
                rho, p_value = spearmanr(group[measure], values.reindex(group.index))
                rows.append(
                    {"horizon": horizon, "model": model, "importance": measure,
                     "reference": reference, "spearman": float(rho), "p_value": float(p_value)}
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Charts and summary
# --------------------------------------------------------------------------------------


def _style(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY)


def _plot_curve(x, y, xlabel: str, ylabel: str, title: str, path: Path, bars=None,
                tick_prefix: str = "") -> None:
    """Line (cumulative) with optional per-component bars; every point and bar is labelled in %."""
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True) * 100
    figure, axis = plt.subplots(figsize=(15, 6), facecolor=SURFACE)
    if bars is not None:
        bars = pd.Series(bars).reset_index(drop=True) * 100
        drawn = axis.bar(x, bars, width=0.7, color="#9ec5f0", label="Per component (% of total)")
        for index, (rect, value) in enumerate(zip(drawn, bars)):
            text = f"{value:.1f}%" if value >= 0.05 else "<0.1%"
            if index == 0:  # first bar top coincides with the first cumulative point
                axis.text(rect.get_x() + rect.get_width() / 2, value / 2, text, ha="center",
                          va="center", fontsize=8, color=TEXT_PRIMARY)
            else:
                axis.text(rect.get_x() + rect.get_width() / 2, value + 0.8, text, ha="center",
                          va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    axis.plot(x, y, color=HORIZON_COLORS["1 hour"], linewidth=2, marker="o", markersize=5,
              label="Cumulative (% of total)" if bars is not None else "Quality (% rebuilt)")
    for xi, yi in zip(x, y):
        axis.annotate(f"{yi:.1f}%", (xi, yi), xytext=(0, 7), textcoords="offset points",
                      ha="center", fontsize=7.5, color=TEXT_PRIMARY, zorder=5,
                      bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"})
    target = VARIANCE_TARGET * 100
    axis.axhline(target, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    axis.text(x.iloc[0] - 0.4, target + 1.2, "95% threshold", ha="left", fontsize=9,
              color=TEXT_SECONDARY)
    reached = y >= target
    if reached.any():
        position = int(reached.idxmax())
        name = f"{tick_prefix}{int(x[position])}" if tick_prefix else f"rank {int(x[position])}"
        axis.annotate(f"95% reached at {name} ({y[position]:.1f}%)", (x[position], y[position]),
                      xytext=(25, -45), textcoords="offset points", fontsize=9,
                      color=TEXT_PRIMARY, arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY})
    axis.set_xticks(list(x), [f"{tick_prefix}{int(value)}" for value in x],
                    fontsize=8 if tick_prefix else 9)
    axis.set_ylim(0, 108)
    axis.set_xlabel(xlabel, color=TEXT_SECONDARY)
    axis.set_ylabel(ylabel, color=TEXT_SECONDARY)
    axis.set_title(title, color=TEXT_PRIMARY, loc="left")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="center right")
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(figure)


def _rank_table(importance: pd.DataFrame, variance: pd.DataFrame, labels: list[str]):
    ranks = pd.DataFrame(index=variance.index)
    ranks["PCA"] = variance["pca"].rank(ascending=False, method="min")
    ranks["NNMF"] = variance["nnmf"].rank(ascending=False, method="min")
    for model in TREE_MODELS:
        short = "Tree" if model == "decision_tree" else "Forest"
        for label in labels:
            values = importance[
                (importance["model"] == model) & (importance["horizon"] == label)
            ].set_index("feature")["permutation_mae_increase"]
            ranks[f"{short}\n{label}"] = values.rank(ascending=False, method="min")
    return ranks.sort_values("PCA")


def _plot_rank_heatmap(importance, variance, labels, output: Path) -> None:
    ranks = _rank_table(importance, variance, labels)
    figure, axis = plt.subplots(figsize=(11, 10), facecolor=SURFACE)
    image = axis.imshow(ranks.to_numpy(), cmap="Blues_r", aspect="auto")
    for (row, column), value in np.ndenumerate(ranks.to_numpy()):
        shade = value <= len(ranks) * 0.35
        axis.text(column, row, f"{int(value)}", ha="center", va="center", fontsize=8,
                  color="white" if shade else TEXT_PRIMARY)
    axis.set_xticks(range(ranks.shape[1]), ranks.columns, fontsize=9, color=TEXT_PRIMARY)
    axis.set_yticks(range(len(ranks)), ranks.index, fontsize=9, color=TEXT_PRIMARY)
    axis.axvline(1.5, color=SURFACE, linewidth=4)
    axis.axvline(1.5 + len(labels), color=SURFACE, linewidth=4)
    axis.tick_params(length=0)
    axis.set_xlabel("Ranking source (variance method, or model and forecast horizon)",
                    color=TEXT_SECONDARY)
    axis.set_ylabel("Feature (model input)", color=TEXT_SECONDARY)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("Rank (1 = most variance captured / largest MAE increase when shuffled)",
                       color=TEXT_SECONDARY)
    colorbar.ax.invert_yaxis()
    axis.set_title(
        "Feature rank: variance captured (PCA, NNMF) vs permutation importance by horizon",
        color=TEXT_PRIMARY, loc="left", fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output / "rank_heatmap.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def _grouped_bars(axis, frame: pd.DataFrame, labels: list[str], value: str, groups: list[str],
                  group_names: dict[str, str], fmt: str) -> None:
    width = 0.8 / len(labels)
    x = np.arange(len(groups))
    for position, label in enumerate(labels):
        values = [frame[(frame["group"] == group) & (frame["horizon"] == label)][value].iloc[0]
                  for group in groups]
        bars = axis.bar(x + (position - (len(labels) - 1) / 2) * width, values, width * 0.92,
                        color=HORIZON_COLORS[label], label=label)
        axis.bar_label(bars, fmt=fmt, padding=2, fontsize=8, color=TEXT_SECONDARY)
    axis.set_xticks(x, [group_names[group] for group in groups], color=TEXT_PRIMARY)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    _style(axis)


def _plot_agreement(agreement: pd.DataFrame, labels: list[str], output: Path) -> None:
    subset = agreement[agreement["importance"] == "permutation_mae_increase"].copy()
    names = {"pca": "PCA", "nnmf": "NNMF", "target_pca": "PCA with target"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, facecolor=SURFACE)
    for axis, model in zip(axes, TREE_MODELS):
        frame = subset[subset["model"] == model].rename(columns={"reference": "group"})
        _grouped_bars(axis, frame, labels, "spearman", list(names), names, "%.2f")
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=1)
        axis.set_title(model.replace("_", " ").title(), color=TEXT_PRIMARY, loc="left")
        axis.set_xlabel("Variance ranking compared with permutation importance",
                        color=TEXT_SECONDARY)
    axes[0].set_ylabel("Spearman rank correlation (unitless, -1 to 1)", color=TEXT_SECONDARY)
    axes[1].legend(title="Forecast horizon", frameon=False)
    figure.suptitle("Agreement between variance ranking and permutation importance",
                    color=TEXT_PRIMARY, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(output / "rank_agreement.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def _plot_mae(metrics: pd.DataFrame, labels: list[str], output: Path) -> None:
    frame = metrics[metrics["feature_set"] == "weather_enabled"].rename(columns={"model": "group"})
    names = {"persistence": "Persistence", "moving_average_24h": "24-hour average",
             "decision_tree": "Decision Tree", "random_forest": "Random Forest"}
    figure, axis = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _grouped_bars(axis, frame, labels, "mae", list(names), names, "%.0f")
    axis.set_ylabel("Holdout MAE (MW, lower is better)", color=TEXT_SECONDARY)
    axis.set_xlabel("Model (each at its cross-validated settings)", color=TEXT_SECONDARY)
    axis.legend(title="Forecast horizon", frameon=False)
    axis.set_title("Holdout error by forecast horizon (energy + weather)",
                   color=TEXT_PRIMARY, loc="left")
    figure.tight_layout()
    figure.savefig(output / "mae_by_horizon.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def _plot_depth_curve(depth_curve: pd.DataFrame, labels: list[str], output: Path) -> None:
    """Training vs holdout MAE by max depth, one panel per horizon (overfitting check)."""
    figure, axes = plt.subplots(1, len(labels), figsize=(18, 6), facecolor=SURFACE)
    for axis, label in zip(np.atleast_1d(axes), labels):
        frame = depth_curve[depth_curve["horizon"] == label].reset_index(drop=True)
        x = np.arange(len(frame))
        axis.plot(x, frame["train_mae_mw"], color=TEXT_SECONDARY, linewidth=2, marker="o",
                  markersize=5, linestyle="--", label="Training MAE (MW)")
        axis.plot(x, frame["holdout_mae_mw"], color=HORIZON_COLORS[label], linewidth=2,
                  marker="o", markersize=5, label="Holdout MAE (MW)")
        for xi, train_value, holdout_value in zip(
            x, frame["train_mae_mw"], frame["holdout_mae_mw"]
        ):
            # Higher value labelled above its point, lower value below, so crossing lines
            # never put two labels in the same place.
            holdout_up = holdout_value >= train_value
            axis.annotate(f"{holdout_value:,.0f}", (xi, holdout_value),
                          xytext=(0, 6 if holdout_up else -11), textcoords="offset points",
                          ha="center", fontsize=6.5, color=TEXT_PRIMARY)
            axis.annotate(f"{train_value:,.0f}", (xi, train_value),
                          xytext=(0, -11 if holdout_up else 6), textcoords="offset points",
                          ha="center", fontsize=6.5, color=TEXT_SECONDARY)
        selected = str(frame["selected_depth"].iloc[0] or "none")
        position = frame.index[frame["max_depth"].astype(str) == selected]
        if len(position):
            axis.axvline(position[0], color=TEXT_PRIMARY, linewidth=1, linestyle=":")
            axis.annotate(f"CV choice: depth {selected}", (position[0], axis.get_ylim()[1]),
                          xytext=(4, -14), textcoords="offset points", fontsize=9,
                          color=TEXT_PRIMARY)
        axis.set_xticks(x, frame["max_depth"].astype(str), fontsize=8)
        axis.set_xlabel("Maximum tree depth (levels)", color=TEXT_SECONDARY)
        axis.set_title(
            f"{label} ahead (min samples per leaf = {frame['min_samples_leaf'].iloc[0]})",
            color=TEXT_PRIMARY, loc="left",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        _style(axis)
    for index, axis in enumerate(np.atleast_1d(axes)):
        # Every panel gets its own legend: the holdout line colour changes with the horizon.
        axis.set_ylabel("MAE (MW, lower is better)", color=TEXT_SECONDARY)
        axis.legend(frameon=False, loc="lower left" if index == 0 else "center right")
    figure.suptitle("Decision tree overfitting check: training vs holdout error by depth",
                    color=TEXT_PRIMARY, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(output / "tree_depth_curve.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, divider, *rows])


def _write_summary(output, metrics, importance, variance, variance_info, agreement, labels):
    table = metrics.copy()
    table["mae"] = table["mae"].map(lambda value: f"{value:,.1f}")
    table["mape_percent"] = table["mape_percent"].map(lambda value: f"{value:.2f}%")
    table = table[["horizon", "feature_set", "model", "mae", "mape_percent"]]

    top_rows = []
    for (horizon, model), group in importance.groupby(["horizon", "model"], sort=False):
        top = group.nlargest(5, "permutation_mae_increase")["feature"].tolist()
        top_rows.append({"horizon": horizon, "model": model, "top 5 (permutation)": ", ".join(top)})

    agreement_table = agreement[agreement["importance"] == "permutation_mae_increase"].pivot_table(
        index=["model", "horizon"], columns="reference", values="spearman", sort=False
    ).round(2).reset_index()

    text = f"""# Feature importance across forecast horizons

Row t keeps its features (data through t - 1) and predicts load at t + horizon - 1:
{", ".join(f"{label} ahead = shift {h - 1} h" for label, h in HORIZONS.items() if label in labels)}.
All horizons share one split time; training rows whose target lands in the test period are
dropped, and tuning folds use a gap equal to the shift.

## Variance side (PCA and NNMF on the model feature table)

- PCA: {variance_info['pca']['components_kept']} components keep
  {variance_info['pca']['variance_explained']:.1%} of variance (PC1 alone
  {variance_info['pca']['pc1_variance']:.1%}).
- NNMF: rank {variance_info['nnmf']['rank']} reaches
  {variance_info['nnmf']['reconstruction_quality']:.1%} reconstruction quality.
- Top features by PCA variance captured: {", ".join(variance['pca'].nlargest(5).index)}.
- Top features by NNMF variance captured: {", ".join(variance['nnmf'].nlargest(5).index)}.

## Holdout error

{_markdown_table(table)}

## Most important features (permutation importance, energy + weather)

{_markdown_table(pd.DataFrame(top_rows))}

## Rank agreement with the variance rankings (Spearman, permutation importance)

{_markdown_table(agreement_table)}

## Notes

- `is_weekend` and `is_holiday` describe the forecast hour, not the target hour.
- `hour` and `day_of_week` are numeric codes, so PCA/NNMF treat 23:00 and 00:00 as far apart.
- Weather keeps its one-hour lag; at every horizon it only uses weather observed before the
  forecast time.
"""
    (output / "summary.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/PJME_hourly_clean.csv"))
    parser.add_argument(
        "--weather", type=Path, default=Path("data/processed/weather_features.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/horizons"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(
        args.data,
        args.weather,
        args.output,
        test_fraction=args.test_fraction,
        cv_splits=args.cv_splits,
        permutation_repeats=args.permutation_repeats,
    )
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
