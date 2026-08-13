from __future__ import annotations

import argparse
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CLASSICAL = r"outputs_classical_v3\per_image.csv"
DEFAULT_DL = r"outputs_dl\per_image.csv"
DEFAULT_OUT = r"outputs_compare_v3"


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    """
    Robust correlation:
      - coercion to numeric
      - drop NaNs
      - Spearman computed as Pearson correlation on ranks (no SciPy dependency)
      - returns NaN if insufficient data or constant series
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return float("nan")
    if a.nunique() <= 1 or b.nunique() <= 1:
        return float("nan")

    if method == "pearson":
        return float(a.corr(b, method="pearson"))

    if method == "spearman":
        ar = a.rank(method="average")
        br = b.rank(method="average")
        return float(ar.corr(br, method="pearson"))

    return float("nan")


def mae(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() == 0:
        return float("nan")
    return float((a[m] - b[m]).abs().mean())


def join_key_from_path(path_str: str, mode: str) -> str:
    p = Path(str(path_str))
    if mode == "fullpath":
        # Normalize slashes to be consistent across CSVs
        return str(p).replace("\\", "/")
    # basename
    return p.name


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "file" not in df.columns:
        raise ValueError(f"CSV missing required column 'file': {path}")
    return df


def plot_scatter(out_dir: Path, metric: str, merged: pd.DataFrame) -> None:
    c = f"{metric}_classical"
    d = f"{metric}_dl"
    m = merged[[c, d]].copy()
    m[c] = pd.to_numeric(m[c], errors="coerce")
    m[d] = pd.to_numeric(m[d], errors="coerce")
    m = m.dropna()
    if m.empty:
        return

    plt.figure()
    plt.scatter(m[c], m[d])
    plt.xlabel(f"{metric} (classical)")
    plt.ylabel(f"{metric} (DL)")
    plt.title(f"Scatter: {metric} (classical vs DL)")
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_{metric}.png", dpi=200)
    plt.close()


def plot_hist(out_dir: Path, metric: str, merged: pd.DataFrame) -> None:
    c = f"{metric}_classical"
    d = f"{metric}_dl"
    m = merged[[c, d]].copy()
    m[c] = pd.to_numeric(m[c], errors="coerce")
    m[d] = pd.to_numeric(m[d], errors="coerce")
    m = m.dropna()
    if m.empty:
        return

    plt.figure()
    plt.hist(m[c], bins=15, alpha=0.7, label="classical")
    plt.hist(m[d], bins=15, alpha=0.7, label="DL")
    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Histogram: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_{metric}.png", dpi=200)
    plt.close()


def plot_trend(out_dir: Path, metric: str, merged: pd.DataFrame) -> None:
    """
    Plot mean by year with simple SEM error bars.
    """
    if "year" not in merged.columns:
        return

    c = f"{metric}_classical"
    d = f"{metric}_dl"
    cols = ["year", c, d]
    m = merged[cols].copy()
    m[c] = pd.to_numeric(m[c], errors="coerce")
    m[d] = pd.to_numeric(m[d], errors="coerce")
    m = m.dropna(subset=["year"])
    if m.empty:
        return

    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m = m.dropna(subset=["year"])
    if m.empty:
        return
    m["year"] = m["year"].astype(int)

    g = m.groupby("year").agg(
        classical_mean=(c, "mean"),
        classical_std=(c, "std"),
        classical_n=(c, "count"),
        dl_mean=(d, "mean"),
        dl_std=(d, "std"),
        dl_n=(d, "count"),
    ).reset_index()

    years = sorted(g["year"].tolist())

    def sem(std, n):
        if pd.isna(std) or n is None or n < 2:
            return 0.0
        return float(std) / math.sqrt(float(n))

    plt.figure()
    plt.plot(years, g.set_index("year").loc[years, "classical_mean"], marker="o", label="classical")
    plt.plot(years, g.set_index("year").loc[years, "dl_mean"], marker="o", label="DL")

    classical_y = g.set_index("year").loc[years, "classical_mean"]
    classical_err = [
        sem(g.set_index("year").loc[y, "classical_std"], g.set_index("year").loc[y, "classical_n"])
        for y in years
    ]
    dl_y = g.set_index("year").loc[years, "dl_mean"]
    dl_err = [
        sem(g.set_index("year").loc[y, "dl_std"], g.set_index("year").loc[y, "dl_n"])
        for y in years
    ]

    plt.errorbar(years, classical_y, yerr=classical_err, fmt="none", capsize=3)
    plt.errorbar(years, dl_y, yerr=dl_err, fmt="none", capsize=3)

    plt.title(f"Trend by year: {metric}")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"trend_{metric}.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare classical vs deep-learning per_image.csv outputs.")
    ap.add_argument("--classical_csv", default=DEFAULT_CLASSICAL, help="Path to classical per_image.csv")
    ap.add_argument("--dl_csv", default=DEFAULT_DL, help="Path to DL per_image.csv")
    ap.add_argument("--out_dir", default=DEFAULT_OUT, help="Output directory for merged CSVs and figures")
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=["vegetation_frac", "water_frac"],
        help="Metrics to compare (must exist in both CSVs)",
    )
    ap.add_argument(
        "--join_mode",
        choices=["basename", "fullpath"],
        default="basename",
        help="How to join images across CSVs. 'basename' is most robust.",
    )
    args = ap.parse_args()

    classical_csv = args.classical_csv
    dl_csv = args.dl_csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using:")
    print(f"  classical_csv = {classical_csv}")
    print(f"  dl_csv        = {dl_csv}")
    print(f"  out_dir       = {out_dir}")

    df_c = load_csv(classical_csv).copy()
    df_d = load_csv(dl_csv).copy()

    df_c["join_key"] = df_c["file"].apply(lambda s: join_key_from_path(s, args.join_mode))
    df_d["join_key"] = df_d["file"].apply(lambda s: join_key_from_path(s, args.join_mode))

    # Keep common metadata columns (prefer classical year/datetime if present)
    keep_c = ["file", "join_key"]
    keep_d = ["file", "join_key"]

    if "year" in df_c.columns:
        keep_c.append("year")
    elif "year" in df_d.columns:
        keep_d.append("year")

    # Build metric columns with suffixes
    for metric in args.metrics:
        if metric not in df_c.columns:
            raise ValueError(f"Metric '{metric}' not found in classical CSV columns.")
        if metric not in df_d.columns:
            raise ValueError(f"Metric '{metric}' not found in DL CSV columns.")

    df_c_small = df_c[keep_c + args.metrics].rename(
        columns={m: f"{m}_classical" for m in args.metrics}
    )
    df_d_small = df_d[keep_d + args.metrics].rename(
        columns={m: f"{m}_dl" for m in args.metrics}
    )

    merged = pd.merge(df_c_small, df_d_small, on="join_key", how="inner", suffixes=("_c", "_d"))

    # Choose a file path column to keep for convenience
    if "file_x" in merged.columns and "file_y" in merged.columns:
        merged = merged.rename(columns={"file_x": "file_classical", "file_y": "file_dl"})
    elif "file" in merged.columns:
        merged = merged.rename(columns={"file": "file_classical"})

    # Prefer classical year if present
    if "year_x" in merged.columns and "year_y" in merged.columns:
        merged = merged.rename(columns={"year_x": "year_classical", "year_y": "year_dl"})
        merged["year"] = pd.to_numeric(merged["year_classical"], errors="coerce").fillna(
            pd.to_numeric(merged["year_dl"], errors="coerce")
        )
    elif "year" in merged.columns:
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce")

    merged_path = out_dir / "merged_per_image.csv"
    merged.to_csv(merged_path, index=False)

    # Summary metrics
    rows = []
    for metric in args.metrics:
        c = f"{metric}_classical"
        d = f"{metric}_dl"
        pear = safe_corr(merged[c], merged[d], "pearson")
        spear = safe_corr(merged[c], merged[d], "spearman")
        err = mae(merged[c], merged[d])
        cmean = float(pd.to_numeric(merged[c], errors="coerce").mean())
        dmean = float(pd.to_numeric(merged[d], errors="coerce").mean())
        n = int(pd.to_numeric(merged[c], errors="coerce").notna().sum())
        rows.append(
            {
                "metric": metric,
                "n": n,
                "pearson_corr": pear,
                "spearman_corr": spear,
                "mae": err,
                "classical_mean": cmean,
                "dl_mean": dmean,
            }
        )

    summary = pd.DataFrame(rows).sort_values(by="spearman_corr", ascending=False, na_position="last")
    summary_path = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Year means compare
    if "year" in merged.columns and merged["year"].notna().any():
        ym_rows = []
        tmp = merged.copy()
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp = tmp.dropna(subset=["year"])
        if not tmp.empty:
            tmp["year"] = tmp["year"].astype(int)
            for metric in args.metrics:
                c = f"{metric}_classical"
                d = f"{metric}_dl"
                g = tmp.groupby("year").agg(
                    classical_mean=(c, "mean"),
                    dl_mean=(d, "mean"),
                    classical_n=(c, "count"),
                    dl_n=(d, "count"),
                )
                g = g.reset_index()
                g.insert(1, "metric", metric)
                ym_rows.append(g)

            if ym_rows:
                year_means = pd.concat(ym_rows, ignore_index=True)
                year_means_path = out_dir / "year_means_compare.csv"
                year_means.to_csv(year_means_path, index=False)
            else:
                year_means_path = None
        else:
            year_means_path = None
    else:
        year_means_path = None

    # Figures
    for metric in args.metrics:
        plot_scatter(out_dir, metric, merged)
        plot_hist(out_dir, metric, merged)
        plot_trend(out_dir, metric, merged)

    print("Wrote:")
    print(f"  {merged_path}")
    print(f"  {summary_path}")
    if year_means_path:
        print(f"  {year_means_path}")
    print("  outputs_compare_v3\\scatter_*.png")
    print("  outputs_compare_v3\\hist_*.png")
    print("  outputs_compare_v3\\trend_*.png")
    print(f"Merged images: {len(merged)}")

    print("\nTop-line summary (sorted by Spearman):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
