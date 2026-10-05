from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



CSV_PATH = "../../../data/processed/PJME_hourly_clean.csv"   #The path
OUT_DIR = Path("../../../reports")
VALUE_COL = "PJME_MW"
PROGRESS_FLAGS = True
STANDARDIZE = True           # True = z-score each hour column (correlation PCA); False = center only
VARIANCE_TARGET = 0.95       # keep enough PCs to explain this much variance
ALPHA = 0.99                 
SHOW_PLOTS = True           # True to also display figures interactively

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
    """
    Closes all plots

    Args:
        name (str): the name of the file
    """
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", dpi=150)
 
    plt.close("all")


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the given data from the path

    Args:
        path (str): the path to the data

    Returns:
        pd.DataFrame: a data frame to use
    """
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


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar columns (hour, day of week, month, season, day type)
    to a copy of the hourly data for EDA plotting.
    """
    d = df[[VALUE_COL]].copy()
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["year"] = d.index.year
    d["is_weekend"] = d["dow"] >= 5
    d["day_type"] = np.where(d["is_weekend"], "Weekend", "Weekday")
    d["season"] = d["month"].map(SEASON_MAP)
    return d


def plot_temporal_patterns(d: pd.DataFrame) -> None:
    daily = d[VALUE_COL].resample("D").mean().to_frame("daily_mean")
    daily["dow"] = daily.index.dayofweek
    daily["month"] = daily.index.month

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Hourly: mean +/- 1 SD across all days
    sns.lineplot(data=d, x="hour", y=VALUE_COL, errorbar="sd", marker="o", ax=axes[0, 0])
    axes[0, 0].set(title="Hourly pattern (mean ± 1 SD)", xlabel="Hour of day",
                   ylabel="MW", xticks=range(0, 24, 2))

    # Daily: distribution of daily mean load by day of week
    sns.boxplot(data=daily, x="dow", y="daily_mean", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set(title="Daily pattern by day of week", xlabel="", ylabel="Daily mean MW")
    axes[0, 1].set_xticks(range(7), DAY_NAMES)

    # Monthly: seasonal cycle
    sns.boxplot(data=daily, x="month", y="daily_mean", ax=axes[1, 0], color="seagreen")
    axes[1, 0].set(title="Monthly (seasonal) pattern", xlabel="", ylabel="Daily mean MW")
    axes[1, 0].set_xticks(range(12), MONTH_NAMES)

    # Annual cycle, one line per year
    pivot = (daily.assign(year=daily.index.year)
                  .groupby(["year", "month"])["daily_mean"].mean().unstack("year"))
    pivot.plot(ax=axes[1, 1], lw=1, legend=False, colormap="viridis")
    axes[1, 1].set(title="Monthly mean by year (each line = one year)",
                   xlabel="", ylabel="MW")
    axes[1, 1].set_xticks(range(1, 13), MONTH_NAMES, rotation=45)
    finish_plot("eda_01_hourly_daily_monthly")

    # Weekly and monthly time series over the full record
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    d[VALUE_COL].resample("W").mean().plot(ax=axes[0], lw=0.8)
    axes[0].set(title="Weekly mean load", ylabel="MW", xlabel="")
    d[VALUE_COL].resample("MS").mean().plot(ax=axes[1], lw=1.2, color="crimson")
    axes[1].set(title="Monthly mean load", ylabel="MW", xlabel="")
    finish_plot("eda_02_weekly_monthly_trend")


def plot_dow_hour_heatmap(d: pd.DataFrame) -> None:
    hm = d.pivot_table(index="dow", columns="hour", values=VALUE_COL, aggfunc="mean")
    hm.index = DAY_NAMES
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(hm, cmap="YlOrRd", cbar_kws={"label": "Mean MW"}, ax=ax)
    ax.set(title="Mean load: day of week × hour of day", xlabel="Hour", ylabel="")
    finish_plot("eda_03_dow_hour_heatmap")



def weekday_weekend_analysis(d: pd.DataFrame) -> None:
    # Overall hourly profiles + weekday-weekend gap by season
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=d, x="hour", y=VALUE_COL, hue="day_type", errorbar="sd",
                 marker="o", ax=axes[0])
    axes[0].set(title="Hourly profile: weekday vs weekend", xlabel="Hour of day",
                ylabel="MW", xticks=range(0, 24, 2))

    prof = (d.groupby(["season", "day_type", "hour"])[VALUE_COL].mean()
              .unstack("day_type"))
    prof["diff"] = prof["Weekday"] - prof["Weekend"]
    diff = prof["diff"].reset_index()
    sns.lineplot(data=diff, x="hour", y="diff", hue="season", marker="o", ax=axes[1])
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(title="Weekday − weekend gap by hour and season",
                xlabel="Hour of day", ylabel="MW difference",
                xticks=range(0, 24, 2))
    finish_plot("eda_04_weekday_vs_weekend")

    # Seasonal facets
    sns.relplot(data=d, x="hour", y=VALUE_COL, hue="day_type", col="season",
                col_order=["Winter", "Spring", "Summer", "Fall"], kind="line",
                errorbar=None, height=3.4, aspect=1.0).set_titles("{col_name}")
    finish_plot("eda_05_weekend_by_season")

    # Does day type matter? Tests on daily mean load
    daily = d.groupby([d.index.normalize(), "day_type"])[VALUE_COL].mean().reset_index()
    daily.columns = ["date", "day_type", "mean_mw"]
    wd = daily.loc[daily["day_type"] == "Weekday", "mean_mw"]
    we = daily.loc[daily["day_type"] == "Weekend", "mean_mw"]

    t, p_t = stats.ttest_ind(wd, we, equal_var=False)  # Welch
    u, p_u = stats.mannwhitneyu(wd, we, alternative="two-sided")
    pooled_sd = np.sqrt((wd.var(ddof=1) + we.var(ddof=1)) / 2)
    cohen_d = (wd.mean() - we.mean()) / pooled_sd

    print("\n=== Weekday vs weekend (daily mean load) ===")
    print(f"Weekday mean: {wd.mean():,.0f} MW (n={len(wd)})")
    print(f"Weekend mean: {we.mean():,.0f} MW (n={len(we)})")
    print(f"Difference:   {wd.mean() - we.mean():,.0f} MW "
          f"({(wd.mean() / we.mean() - 1):.1%})")
    print(f"Welch t-test: t={t:.2f}, p={p_t:.3g}")
    print(f"Mann-Whitney: U={u:,.0f}, p={p_u:.3g}")
    print(f"Cohen's d:    {cohen_d:.2f}")

    # Across all seven days of the week
    daily["dow"] = daily["date"].dt.dayofweek
    groups = [g["mean_mw"].values for _, g in daily.groupby("dow")]
    f_stat, p_f = stats.f_oneway(*groups)
    h_stat, p_h = stats.kruskal(*groups)
    print(f"\nAcross all 7 days -> ANOVA F={f_stat:.1f}, p={p_f:.3g}; "
          f"Kruskal H={h_stat:.1f}, p={p_h:.3g}")

    summary = daily.groupby("dow")["mean_mw"].agg(["mean", "std", "count"])
    summary.index = DAY_NAMES
    print("\nDaily mean load by day of week:")
    print(summary.round(0))

    summary.to_csv(OUT_DIR / "eda_dow_summary.csv")
    prof.reset_index().to_csv(OUT_DIR / "eda_weekday_weekend_hourly.csv", index=False)


def run_eda(df: pd.DataFrame) -> None:
    """
    runs all eda analysis

    Args:
        df (pd.DataFrame): the given data file
    """
    d = add_calendar(df)
    plot_temporal_patterns(d)
    plot_dow_hour_heatmap(d)
    weekday_weekend_analysis(d)



def build_daily_matrix(df: pd.DataFrame):
    """
    Builds a matrix that is a hourly representation of a day

    Args:
        df (pd.DataFrame): the data frame from load_data

    Returns:
        matrix: _description_
    """
    d = df.copy()
    d["date"] = d.index.normalize()
    d["hour"] = d.index.hour

    # mean() collapses DST duplicate hours into one value
    X = d.pivot_table(index="date", columns="hour", values=VALUE_COL, aggfunc="mean")
    day_flags = d.groupby("date").agg(
        imputed_frac=("is_imputed", "mean"),
        has_outlier=("is_outlier", "any"),
    )

    # Keep only complete 24-hour days (DST spring-forward day loses an hour)
    complete = X.notna().all(axis=1)
    print(f"Days total: {len(X):,} | incomplete (dropped): {(~complete).sum()}")
    X, day_flags = X[complete], day_flags.loc[complete]



    # Day-level metadata used for colouring and regression (not fed into PCA)
    meta = day_flags.copy()
    meta["daily_mean"] = X.mean(axis=1)
    meta["weekday"] = X.index.dayofweek
    meta["is_weekend"] = meta["weekday"] >= 5
    meta["month"] = X.index.month
    meta["year"] = X.index.year
    meta["season"] = meta["month"].map(SEASON_MAP)


    X.columns = [f"h{h:02d}" for h in X.columns]
    return X, meta



def fit_pca(X: pd.DataFrame):
    scaler = StandardScaler(with_std=STANDARDIZE)
    Z = scaler.fit_transform(X)

    pca = PCA(n_components=None, random_state=0).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, VARIANCE_TARGET) + 1)
    print(f"\nPCs needed for {VARIANCE_TARGET:.0%} variance: {k}")
    print("Explained variance (first 5 PCs):",
          np.round(pca.explained_variance_ratio_[:5], 4))

    pcs = [f"PC{i+1}" for i in range(Z.shape[1])]
    scores = pd.DataFrame(pca.transform(Z), index=X.index, columns=pcs)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=pcs)
    return Z, pca, k, scores, loadings



def plot_scree(pca: PCA, k: int) -> None:
    evr = pca.explained_variance_ratio_[:12]
    cum = np.cumsum(pca.explained_variance_ratio_)[:12]
    x = np.arange(1, len(evr) + 1)
    _, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, evr, alpha=0.7, label="Individual")
    ax.plot(x, cum, "o-", color="crimson", label="Cumulative")
    ax.axhline(VARIANCE_TARGET, ls="--", color="gray")
    ax.axvline(k, ls=":", color="gray")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio",
           title=f"Scree plot (k={k} retained)")
    ax.legend()
    finish_plot("01_scree")


def plot_loadings(loadings: pd.DataFrame, n: int = 4) -> None:
    cols = loadings.columns[:n]
    _, ax = plt.subplots(figsize=(9, 4.5))
    for c in cols:
        ax.plot(range(24), loadings[c].values, marker="o", ms=3, label=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Hour of day", ylabel="Loading",
           title="PC loadings across the day", xticks=range(0, 24, 2))
    ax.legend()
    finish_plot("02_loadings_lines")

    _, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(loadings[cols], cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                cbar_kws={"label": "Loading"}, ax=ax)
    ax.set_title("Loadings heatmap")
    finish_plot("03_loadings_heatmap")


def plot_scores(scores: pd.DataFrame, meta: pd.DataFrame) -> None:
    _, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x=scores["PC1"], y=scores["PC2"], hue=meta["season"],
                    s=12, alpha=0.6, ax=axes[0])
    axes[0].set_title("Day scores by season")
    sns.scatterplot(x=scores["PC1"], y=scores["PC2"], hue=meta["is_weekend"],
                    s=12, alpha=0.6, ax=axes[1])
    axes[1].set_title("Day scores by weekend")
    finish_plot("04_score_scatter")

    _, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, pc in zip(axes, ["PC1", "PC2", "PC3"]):
        ax.plot(scores.index, scores[pc], lw=0.4, alpha=0.6)
        ax.plot(scores.index, scores[pc].rolling(30, center=True).mean(),
                color="crimson", lw=1.2, label="30-day mean")
        ax.set_ylabel(pc)
    axes[0].legend(loc="upper right")
    axes[0].set_title("PC scores over time")
    finish_plot("05_scores_timeseries")



def main() -> None:
    df = load_data(CSV_PATH)

    # Visual EDA + weekday/weekend comparison (before any modelling)
    run_eda(df)

    X, meta = build_daily_matrix(df)
    print(f"Daily matrix shape: {X.shape}")

    _, pca, k, scores, loadings = fit_pca(X)

    plot_scree(pca, k)
    plot_loadings(loadings)
    plot_scores(scores, meta)


    scores.join(meta).to_csv(OUT_DIR / "pc_scores.csv")
    loadings.to_csv(OUT_DIR / "pc_loadings.csv")
    print(f"\nDone. Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()