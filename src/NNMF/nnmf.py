from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "processed" / "PJME_hourly_clean.csv"
OUT_DIR = REPO_ROOT / "reports" / "nnmf"
VALUE_COL = "PJME_MW"
RECONSTRUCTION_TARGET = 0.95
MAX_COMPONENTS = 24
RANDOM_STATE = 0
MAX_ITER = 2_000

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}

sns.set_theme(style="whitegrid", context="notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def finish_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=150)
    plt.close("all")


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()
    for col in ("is_imputed", "is_outlier"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
        else:
            df[col] = False

    print(f"Rows: {len(df):,} | {df.index.min()} -> {df.index.max()}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    print(f"Missing {VALUE_COL}: {df[VALUE_COL].isna().sum()}")
    print(f"Imputed: {df['is_imputed'].mean():.2%} | Outliers: {df['is_outlier'].mean():.2%}")
    return df


def build_daily_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["date"] = d.index.normalize()
    d["hour"] = d.index.hour

    matrix = d.pivot_table(index="date", columns="hour", values=VALUE_COL, aggfunc="mean")
    day_flags = d.groupby("date").agg(
        imputed_frac=("is_imputed", "mean"),
        has_outlier=("is_outlier", "any"),
    )

    complete = matrix.notna().all(axis=1)
    print(f"Days total: {len(matrix):,} | incomplete (dropped): {(~complete).sum()}")
    matrix, day_flags = matrix[complete], day_flags.loc[complete]

    meta = day_flags.copy()
    meta["daily_mean"] = matrix.mean(axis=1)
    meta["weekday"] = matrix.index.dayofweek
    meta["is_weekend"] = meta["weekday"] >= 5
    meta["month"] = matrix.index.month
    meta["year"] = matrix.index.year
    meta["season"] = meta["month"].map(SEASON_MAP)

    matrix.columns = [f"h{hour:02d}" for hour in matrix.columns]
    return matrix, meta


def reconstruction_quality(matrix: np.ndarray, reconstruction: np.ndarray) -> float:
    residual_sum_squares = np.square(matrix - reconstruction).sum()
    baseline_sum_squares = np.square(matrix - matrix.mean()).sum()
    if baseline_sum_squares == 0:
        return 1.0 if residual_sum_squares == 0 else 0.0
    return float(1 - residual_sum_squares / baseline_sum_squares)


def fit_nnmf(
    X: pd.DataFrame,
) -> tuple[NMF, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NNMF input must contain only finite values.")
    if (values < 0).any():
        raise ValueError("NNMF input must be nonnegative.")

    metrics: list[dict[str, float | int]] = []
    selected: tuple[NMF, int, float] | None = None
    for rank in range(1, min(MAX_COMPONENTS, X.shape[1]) + 1):
        model = NMF(
            n_components=rank,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
        )
        scores = model.fit_transform(values)
        reconstruction = model.inverse_transform(scores)
        quality = reconstruction_quality(values, reconstruction)
        metrics.append({
            "rank": rank,
            "reconstruction_error": float(model.reconstruction_err_),
            "reconstruction_quality": quality,
        })
        if quality >= RECONSTRUCTION_TARGET and selected is None:
            selected = model, rank, quality

    metric_frame = pd.DataFrame(metrics)
    if selected is None:
        best_rank = int(metric_frame.loc[metric_frame["reconstruction_quality"].idxmax(), "rank"])
        best_model = NMF(
            n_components=best_rank,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
        )
        best_scores = best_model.fit_transform(values)
        best_quality = reconstruction_quality(values, best_model.inverse_transform(best_scores))
        selected = best_model, best_rank, best_quality
        print(
            f"\nWarning: no rank through {min(MAX_COMPONENTS, X.shape[1])} reached "
            f"{RECONSTRUCTION_TARGET:.0%}; using best rank {best_rank} "
            f"({best_quality:.2%} reconstruction quality)."
        )

    model, rank, _ = selected
    scores = model.transform(values)
    score_columns = [f"Factor{i + 1}" for i in range(rank)]
    score_frame = pd.DataFrame(scores, index=X.index, columns=score_columns)
    components = pd.DataFrame(model.components_.T, index=X.columns, columns=score_columns)
    print(f"\nSelected NNMF rank: {rank}")
    print(metric_frame.head().round(4).to_string(index=False))
    return model, rank, score_frame, components, metric_frame


def plot_rank_metrics(metrics: pd.DataFrame, rank: int) -> None:
    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(metrics["rank"], metrics["reconstruction_quality"], "o-", color="steelblue")
    ax.axhline(RECONSTRUCTION_TARGET, ls="--", color="gray")
    ax.axvline(rank, ls=":", color="crimson")
    ax.set(
        xlabel="Number of factors",
        ylabel="Reconstruction quality",
        title=f"NNMF rank selection (k={rank})",
    )
    ax.set_xticks(metrics["rank"])
    finish_plot("01_rank_selection")


def plot_components(components: pd.DataFrame) -> None:
    hours = range(len(components))
    _, ax = plt.subplots(figsize=(10, 5))
    for column in components.columns:
        ax.plot(hours, components[column], marker="o", ms=3, label=column)
    ax.set(
        xlabel="Hour of day",
        ylabel="Component weight (MW)",
        title="NNMF component profiles",
        xticks=range(0, 24, 2),
    )
    ax.legend()
    finish_plot("02_component_profiles")

    _, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(components, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Component weight (MW)"})
    ax.set(title="NNMF component profiles by hour", xlabel="Factor", ylabel="Hour")
    finish_plot("03_components_heatmap")


def plot_scores(scores: pd.DataFrame, meta: pd.DataFrame) -> None:
    columns = list(scores.columns)
    if len(columns) >= 2:
        _, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.scatterplot(
            x=scores[columns[0]], y=scores[columns[1]], hue=meta["season"],
            s=12, alpha=0.6, ax=axes[0],
        )
        axes[0].set_title("Day activations by season")
        sns.scatterplot(
            x=scores[columns[0]], y=scores[columns[1]], hue=meta["is_weekend"],
            s=12, alpha=0.6, ax=axes[1],
        )
        axes[1].set_title("Day activations by weekend")
        finish_plot("04_activation_scatter")

    n_plots = min(3, len(columns))
    _, axes = plt.subplots(n_plots, 1, figsize=(13, 3 * n_plots), sharex=True, squeeze=False)
    for ax, column in zip(axes[:, 0], columns[:n_plots]):
        ax.plot(scores.index, scores[column], lw=0.4, alpha=0.6)
        ax.plot(
            scores.index,
            scores[column].rolling(30, center=True).mean(),
            color="crimson",
            lw=1.2,
            label="30-day mean",
        )
        ax.set_ylabel(column)
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_title("NNMF factor activations over time")
    finish_plot("05_activation_timeseries")


def main() -> None:
    df = load_data(CSV_PATH)
    X, meta = build_daily_matrix(df)
    print(f"Daily matrix shape: {X.shape}")

    _, rank, scores, components, metrics = fit_nnmf(X)
    plot_rank_metrics(metrics, rank)
    plot_components(components)
    plot_scores(scores, meta)

    scores.join(meta).to_csv(OUT_DIR / "nnmf_scores.csv")
    components.to_csv(OUT_DIR / "nnmf_components.csv")
    metrics.to_csv(OUT_DIR / "nnmf_rank_metrics.csv", index=False)
    print(f"\nDone. Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
