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
MAX_COMPONENTS = 53
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


def finish_plot(name: str, output_dir: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=150)
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


def build_profile_matrix(
    df: pd.DataFrame,
    horizon: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon not in {"hour", "day", "week"}:
        raise ValueError("horizon must be one of: hour, day, week")

    d = df.copy()
    d["date"] = d.index.normalize()
    if horizon == "hour":
        d["position"] = d.index.hour
        d["profile"] = d["date"]
        d["profile_value"] = d[VALUE_COL]
        expected_positions = range(24)
        column_names = [f"h{position:02d}" for position in expected_positions]
        profile_label = "days"
    elif horizon == "day":
        d["position"] = d.index.dayofweek
        d["profile"] = d["date"] - pd.to_timedelta(d["position"], unit="D")
        d["profile_value"] = d[VALUE_COL]
        expected_positions = range(7)
        column_names = DAY_NAMES
        profile_label = "weeks"
    else:
        d = d.groupby("date").agg(
            profile_value=(VALUE_COL, "mean"),
            is_imputed=("is_imputed", "mean"),
            is_outlier=("is_outlier", "any"),
        ).reset_index()
        d["profile"] = pd.to_datetime(d["date"].dt.year.astype(str) + "-01-01")
        d = d[d["date"].dt.month.ne(2) | d["date"].dt.day.ne(29)]
        d["position"] = d.groupby("profile").cumcount() + 1
        expected_positions = range(1, 366)
        column_names = [f"d{position:03d}" for position in expected_positions]
        profile_label = "calendar years"

    matrix = d.pivot_table(index="profile", columns="position", values="profile_value", aggfunc="mean")
    profile_flags = d.groupby("profile").agg(
        imputed_frac=("is_imputed", "mean"),
        has_outlier=("is_outlier", "any"),
    )

    complete = matrix.reindex(columns=expected_positions).notna().all(axis=1)
    print(f"{profile_label.title()} total: {len(matrix):,} | incomplete (dropped): {(~complete).sum()}")
    matrix = matrix.reindex(columns=expected_positions)[complete]
    profile_flags = profile_flags.loc[matrix.index]

    meta = profile_flags.copy()
    meta["profile_mean"] = matrix.mean(axis=1)
    meta["weekday"] = matrix.index.dayofweek
    meta["is_weekend"] = meta["weekday"] >= 5
    meta["month"] = matrix.index.month
    meta["year"] = matrix.index.year
    meta["season"] = meta["month"].map(SEASON_MAP)

    matrix.columns = column_names
    return matrix, meta


def build_daily_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_profile_matrix(df, "hour")


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
    data_min = values.min()
    data_range = values.max() - data_min
    if data_range == 0:
        raise ValueError("NNMF input must contain variation.")
    normalized_values = (values - data_min) / data_range

    metrics: list[dict[str, float | int]] = []
    selected: tuple[NMF, int, float] | None = None
    max_rank = min(MAX_COMPONENTS, X.shape[0], X.shape[1])
    for rank in range(1, max_rank + 1):
        model = NMF(
            n_components=rank,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
        )
        scores = model.fit_transform(normalized_values)
        reconstruction = data_min + data_range * model.inverse_transform(scores)
        quality = reconstruction_quality(values, reconstruction)
        metrics.append({
            "rank": rank,
            "reconstruction_error": float(model.reconstruction_err_ * data_range),
            "reconstruction_quality": quality,
            "iterations": model.n_iter_,
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
        best_scores = best_model.fit_transform(normalized_values)
        best_reconstruction = data_min + data_range * best_model.inverse_transform(best_scores)
        best_quality = reconstruction_quality(values, best_reconstruction)
        selected = best_model, best_rank, best_quality
        print(
            f"\nWarning: no rank through {max_rank} reached "
            f"{RECONSTRUCTION_TARGET:.0%}; using best rank {best_rank} "
            f"({best_quality:.2%} reconstruction quality)."
        )

    model, rank, _ = selected
    scores = model.transform(normalized_values)
    score_columns = [f"Factor{i + 1}" for i in range(rank)]
    score_frame = pd.DataFrame(scores, index=X.index, columns=score_columns)
    components = pd.DataFrame(
        model.components_.T * data_range,
        index=X.columns,
        columns=score_columns,
    )
    print(f"\nSelected NNMF rank: {rank}")
    print(metric_frame.head().round(4).to_string(index=False))
    return model, rank, score_frame, components, metric_frame


def plot_rank_metrics(metrics: pd.DataFrame, rank: int, output_dir: Path, horizon: str) -> None:
    _, ax_quality = plt.subplots(figsize=(8, 4.5))
    ax_error = ax_quality.twinx()
    quality_line = ax_quality.plot(
        metrics["rank"], metrics["reconstruction_quality"], "o-",
        color="steelblue", label="Reconstruction quality",
    )
    error_line = ax_error.plot(
        metrics["rank"], metrics["reconstruction_error"], "s--",
        color="darkorange", label="Reconstruction error",
    )
    ax_quality.axhline(
        RECONSTRUCTION_TARGET, ls=":", color="gray", label="Target quality",
    )
    ax_quality.axvline(rank, ls="-.", color="crimson", label=f"Selected rank (k={rank})")
    ax_quality.set(
        xlabel="Number of factors",
        ylabel="Reconstruction quality",
        title=f"NNMF {horizon} horizon (selected rank k={rank})",
    )
    ax_error.set_ylabel("Reconstruction error")
    ax_quality.set_xticks(metrics["rank"])
    lines = quality_line + error_line + ax_quality.lines[-2:]
    ax_quality.legend(lines, [line.get_label() for line in lines], loc="center right")
    finish_plot("01_rank_selection", output_dir)


def plot_components(components: pd.DataFrame, output_dir: Path, horizon: str) -> None:
    positions = range(len(components))
    _, ax = plt.subplots(figsize=(10, 5))
    for column in components.columns:
        ax.plot(positions, components[column], marker="o", ms=3, label=column)
    tick_step = max(1, len(components) // 12)
    ax.set(
        xlabel={"hour": "Hour of day", "day": "Day of week", "week": "Day of year"}[horizon],
        ylabel="Component weight (MW)",
        title=f"NNMF {horizon} horizon component profiles",
        xticks=list(positions)[::tick_step],
        xticklabels=components.index[::tick_step],
    )
    ax.legend()
    finish_plot("02_component_profiles", output_dir)

    _, ax = plt.subplots(figsize=(7, max(8, len(components) / 8)))
    sns.heatmap(components, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Component weight (MW)"})
    ax.set(
        title=f"NNMF {horizon} horizon component profiles",
        xlabel="Factor",
        ylabel={"hour": "Hour", "day": "Day", "week": "Day of year"}[horizon],
    )
    finish_plot("03_components_heatmap", output_dir)


def plot_top_factors(
    scores: pd.DataFrame,
    components: pd.DataFrame,
    matrix: pd.DataFrame,
    output_dir: Path,
    horizon: str,
    top_n: int = 3,
    representative_profiles: int = 3,
) -> None:
    ranked = scores.mean().sort_values(ascending=False).head(top_n)
    n_factors = len(ranked)
    _, axes = plt.subplots(
        n_factors, 2, figsize=(14, 4 * n_factors),
        squeeze=False, gridspec_kw={"width_ratios": [1, 1.4]},
    )
    positions = range(len(components))

    for row, factor in enumerate(ranked.index):
        factor_scores = scores[factor]
        selected_profiles = factor_scores.nlargest(representative_profiles).index
        profile_ax, dates_ax = axes[row]

        profile_ax.plot(positions, components[factor], marker="o", ms=3, color="steelblue")
        tick_step = max(1, len(components) // 12)
        profile_ax.set(
            title=f"{factor} profile (mean activation {ranked[factor]:.1f})",
            xlabel={"hour": "Hour of day", "day": "Day of week", "week": "Day of year"}[horizon],
            ylabel="Component weight (MW)",
            xticks=list(positions)[::tick_step],
            xticklabels=components.index[::tick_step],
        )

        for profile in selected_profiles:
            dates_ax.plot(
                positions, matrix.loc[profile].to_numpy(), lw=1.5,
                label=f"{profile:%Y-%m-%d} (activation {factor_scores.loc[profile]:.1f})",
            )
        dates_ax.set(
            title=f"{factor} representative profiles",
            xlabel={"hour": "Hour of day", "day": "Day of week", "week": "Day of year"}[horizon],
            ylabel="Load (MW)",
            xticks=list(positions)[::tick_step],
            xticklabels=components.index[::tick_step],
        )
        dates_ax.legend(fontsize="small")

    finish_plot("06_top_factors", output_dir)


def plot_scores(scores: pd.DataFrame, meta: pd.DataFrame, output_dir: Path, horizon: str) -> None:
    columns = list(scores.columns)
    if len(columns) >= 2:
        _, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.scatterplot(
            x=scores[columns[0]], y=scores[columns[1]], hue=meta["season"],
            s=12, alpha=0.6, ax=axes[0],
        )
        axes[0].set_title(f"{horizon.title()} activations by season")
        sns.scatterplot(
            x=scores[columns[0]], y=scores[columns[1]], hue=meta["is_weekend"],
            s=12, alpha=0.6, ax=axes[1],
        )
        axes[1].set_title(f"{horizon.title()} activations by weekend")
        finish_plot("04_activation_scatter", output_dir)

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
    finish_plot("05_activation_timeseries", output_dir)


def run_horizon(df: pd.DataFrame, horizon: str) -> None:
    output_dir = OUT_DIR / horizon
    output_dir.mkdir(parents=True, exist_ok=True)
    X, meta = build_profile_matrix(df, horizon)
    print(f"{horizon.title()} horizon matrix shape: {X.shape}")

    _, rank, scores, components, metrics = fit_nnmf(X)
    plot_rank_metrics(metrics, rank, output_dir, horizon)
    plot_components(components, output_dir, horizon)
    plot_top_factors(scores, components, X, output_dir, horizon)
    plot_scores(scores, meta, output_dir, horizon)

    scores.join(meta).to_csv(output_dir / "nnmf_scores.csv")
    components.to_csv(output_dir / "nnmf_components.csv")
    metrics.to_csv(output_dir / "nnmf_rank_metrics.csv", index=False)
    pd.DataFrame([{
        "horizon": horizon,
        "profiles": len(X),
        "profile_width": X.shape[1],
        "selected_rank": rank,
        "reconstruction_quality": metrics.loc[
            metrics["rank"].eq(rank), "reconstruction_quality"
        ].iloc[0],
    }]).to_csv(output_dir / "nnmf_summary.csv", index=False)
    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")


def main() -> None:
    df = load_data(CSV_PATH)
    for horizon in ("hour", "day", "week"):
        run_horizon(df, horizon)


if __name__ == "__main__":
    main()
