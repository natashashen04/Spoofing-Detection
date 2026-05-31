#!/usr/bin/env python3
"""
spoof_detection.py
==================
Spoofing detection pipeline for ESZ25 limit-order-book data.

Analytical Framework
--------------------
Modeled on the methodology used in CFTC enforcement proceedings (e.g.,
CFTC v. Navinder Singh Sarao, 2015; CFTC v. JPMorgan, 2020), this pipeline
mirrors the four-step approach applied to identify and characterise market
manipulation by a flagged trader:

  Step 1  ETL        – Load raw order-book events; reconstruct per-order
                       lifecycle records (placement → cancellation / fill).

  Step 2  Features   – Compute CFTC-based spoofing metrics per order:
                         · order_duration_s   : seconds from placement to close
                         · is_cancelled       : whether order was cancelled vs filled
                         · quantity           : order size (contracts)
                       And per trader:
                         · suspicious_rate    : fraction of orders meeting all
                                               CFTC spoofing criteria
                         · large_order_otr   : large-cancelled-orders / filled-orders
                                               (captures the signature inflate-and-pull pattern)

  Step 3  Flag       – Identify the trader whose behaviour is most consistent
                       with CFTC spoofing criteria, using suspicious_rate as the
                       primary discriminant. In actual enforcement, the CFTC
                       identifies this trader through referrals and market
                       surveillance before the economic analysis begins.

  Step 4  Lifecycle  – Compare lifecycle diagrams of suspicious vs. non-suspicious
                       orders for the flagged trader, illustrating how suspicious
                       orders manipulate prices before being pulled.

  Step 5  Compare    – Compare median order duration, order size, and large-order OTR
                       between the flagged trader's suspicious orders and all
                       non-flagged market orders.

  Step 6  Inference  – Mann-Whitney U tests: confirm each metric difference is
                       statistically significant at the 5% level (alpha = 0.05).

Input:   market_data_ESZ25.csv        (produced by generate_market_data.py)
Outputs: spoof_detection_report/
           trader_metrics.csv
           suspicious_orders.csv
           statistical_test_results.csv
           fig1_lifecycle_comparison.png
           fig2_episode_timeline.png
           fig3_metric_comparison.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH  = "market_data_ESZ25.csv"
OUTPUT_DIR = Path("spoof_detection_report")

# CFTC-derived spoofing thresholds
# Source: CFTC Guidance on Disruptive Practices (7 U.S.C. § 6c(a)(5)(C))
DURATION_THRESHOLD_S   = 5.0   # orders cancelled in < 5 s are suspicious
SIZE_QUANTILE          = 0.75  # only top-25% orders (by size) qualify
SUSPICIOUS_RATE_FLAG   = 0.50  # trader-level: > 50% of orders meet all 3 criteria
WINDOW_MIN             = 30    # session window length (minutes) for OTR comparison

ALPHA = 0.05   # significance level for hypothesis tests

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "suspicious":     "#c0392b",   # red
    "non_suspicious": "#2980b9",   # blue
    "market":         "#7f8c8d",   # grey
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – ETL
# ══════════════════════════════════════════════════════════════════════════════

def load_events(path: str) -> pd.DataFrame:
    """Load raw order-book event stream from CSV."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["spoof_episode_id"] = df["spoof_episode_id"].fillna("")
    print(f"  Loaded {len(df):,} events for {df['trader_id'].nunique()} traders.")
    return df


def reconstruct_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct per-order lifecycle records from the event stream.

    Each order is matched from its NEW_ORDER event to its terminal event
    (CANCEL, FILL, or PARTIAL_FILL). Orders still open at session end are
    excluded.

    Derived columns
    ---------------
    order_duration_s : elapsed seconds from placement to cancellation/fill
    is_cancelled     : True when the terminal event is CANCEL
    mid_return       : change in mid-price while the order was live
    """
    new_orders = (
        df[df["event_type"] == "NEW_ORDER"]
        .rename(columns={
            "timestamp": "ts_open",
            "best_bid":  "bid_open",
            "best_ask":  "ask_open",
            "mid_price": "mid_open",
        })
        [[
            "order_id", "ts_open", "trader_id", "side",
            "price", "quantity", "bid_open", "ask_open", "mid_open",
            "is_spoof", "spoof_episode_id",
        ]]
    )

    terminal = (
        df[df["event_type"].isin(["CANCEL", "FILL", "PARTIAL_FILL"])]
        .sort_values("timestamp")
        .groupby("order_id", as_index=False)
        .last()[["order_id", "timestamp", "event_type", "mid_price"]]
        .rename(columns={
            "timestamp":  "ts_close",
            "event_type": "terminal_type",
            "mid_price":  "mid_close",
        })
    )

    orders = new_orders.merge(terminal, on="order_id", how="inner")

    orders["order_duration_s"] = (
        (orders["ts_close"] - orders["ts_open"])
        .dt.total_seconds()
        .clip(lower=0)
    )
    orders["is_cancelled"] = orders["terminal_type"] == "CANCEL"
    orders["mid_return"]   = orders["mid_close"] - orders["mid_open"]

    print(f"  Reconstructed {len(orders):,} order lifecycle records.")
    return orders


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify orders as suspicious / non-suspicious and compute per-trader metrics.

    Order-level suspicious flag (all three conditions must hold):
      1. Order was cancelled (not filled)
      2. Order lived < DURATION_THRESHOLD_S seconds
      3. Order size is in the top (1 - SIZE_QUANTILE) of all orders

    Trader-level metrics:
      suspicious_rate  = suspicious_orders / total_orders
      large_order_otr  = large_cancelled_orders / max(filled_orders, 1)
                         Captures the spoofing pattern: many large orders placed
                         and pulled for each execution — the inflate-and-pull ratio.
    """
    size_threshold = orders["quantity"].quantile(SIZE_QUANTILE)

    orders["is_suspicious"] = (
        orders["is_cancelled"]
        & (orders["order_duration_s"] < DURATION_THRESHOLD_S)
        & (orders["quantity"] >= size_threshold)
    )

    # ── Per-trader aggregation ─────────────────────────────────────────────────
    rows = []
    for tid, g in orders.groupby("trader_id"):
        n          = len(g)
        n_cancel   = int(g["is_cancelled"].sum())
        n_fill     = int((~g["is_cancelled"]).sum())
        n_susp     = int(g["is_suspicious"].sum())
        large_cancel = int(
            (g["is_cancelled"] & (g["quantity"] >= size_threshold)).sum()
        )
        rows.append({
            "trader_id":         tid,
            "n_orders":          n,
            "n_cancelled":       n_cancel,
            "n_filled":          n_fill,
            "n_suspicious":      n_susp,
            "suspicious_rate":   n_susp / n,
            "cancellation_rate": n_cancel / n,
            "large_order_otr":   large_cancel / max(n_fill, 1),
            "otr":               n / max(n_fill, 1),
            "median_duration_s": g["order_duration_s"].median(),
            "median_qty":        g["quantity"].median(),
        })

    trader_stats = pd.DataFrame(rows)
    return orders, trader_stats


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Flagging
# ══════════════════════════════════════════════════════════════════════════════

def identify_flagged_trader(trader_stats: pd.DataFrame) -> str:
    """
    Identify the trader whose activity is most consistent with CFTC spoofing
    criteria, using suspicious_rate as the primary discriminant.

    suspicious_rate > 0.50 means the majority of a trader's orders are:
      · large (top quartile by size)
      · cancelled quickly (< DURATION_THRESHOLD_S seconds)
      · never intended to fill

    In actual enforcement, the CFTC identifies this trader through referrals,
    whistleblower tips, and automated market-surveillance alerts before the
    economic analysis begins. The pipeline then characterises their behaviour.
    """
    flagged = trader_stats[
        trader_stats["suspicious_rate"] >= SUSPICIOUS_RATE_FLAG
    ].sort_values("suspicious_rate", ascending=False)

    if flagged.empty:
        raise ValueError("No trader meets suspicious_rate threshold.")

    target = flagged.iloc[0]["trader_id"]
    row    = flagged.iloc[0]

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  FLAGGED TRADER: {target}")
    print(sep)
    print(f"  Suspicious order rate    : {row['suspicious_rate']:.1%}  "
          f"(threshold >= {SUSPICIOUS_RATE_FLAG:.0%})")
    print(f"  Cancellation rate        : {row['cancellation_rate']:.1%}")
    print(f"  Large-order OTR          : {row['large_order_otr']:.2f}x")
    print(f"  Standard OTR             : {row['otr']:.1f}x")
    print(f"  Median order duration    : {row['median_duration_s']:.2f} s")
    print(f"  Median order size        : {row['median_qty']:.0f} contracts")
    print(sep)
    return target


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Lifecycle Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def fig_lifecycle_comparison(
    orders: pd.DataFrame,
    target_id: str,
    out_dir: Path,
) -> None:
    """
    Figure 1 – Order Lifecycle Comparison (Flagged Trader)

    Left panel : Duration distribution for the flagged trader's suspicious
                 orders (large, fast, cancelled) vs. non-suspicious orders
                 (small profit trades that fill normally).

    Right panel: Mid-price change during each order's lifetime.  Suspicious
                 orders coincide with larger price movements — consistent with
                 using manufactured order-book pressure to shift the market
                 before cancelling.
    """
    target = orders[orders["trader_id"] == target_id]
    susp   = target[target["is_suspicious"]]
    legit  = target[~target["is_suspicious"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Order Lifecycle Comparison — Flagged Trader {target_id}\n"
        "Suspicious vs. Non-Suspicious Orders",
        fontsize=13, fontweight="bold",
    )

    # ── Left: duration distribution ────────────────────────────────────────────
    ax   = axes[0]
    cap  = target["order_duration_s"].quantile(0.95)
    bins = np.linspace(0, cap, 45)

    ax.hist(legit["order_duration_s"].clip(upper=cap), bins=bins,
            alpha=0.65, density=True, color=COLORS["non_suspicious"],
            label="Non-suspicious")
    ax.hist(susp["order_duration_s"].clip(upper=cap),  bins=bins,
            alpha=0.78, density=True, color=COLORS["suspicious"],
            label="Suspicious")
    ax.axvline(DURATION_THRESHOLD_S, color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold ({DURATION_THRESHOLD_S:.0f} s)")
    ax.set_xlabel("Order Duration (seconds)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Order Duration Distribution", fontsize=12)
    ax.legend(fontsize=10)

    # ── Right: mid-price change during order life ──────────────────────────────
    ax = axes[1]
    bp = ax.boxplot(
        [susp["mid_return"].values, legit["mid_return"].values],
        labels=[
            f"Suspicious\n(n={len(susp):,})",
            f"Non-suspicious\n(n={len(legit):,})",
        ],
        patch_artist=True, showfliers=False, widths=0.5,
        medianprops=dict(color="white", linewidth=2.5),
    )
    bp["boxes"][0].set_facecolor(COLORS["suspicious"])
    bp["boxes"][1].set_facecolor(COLORS["non_suspicious"])
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Mid-price Change While Order Live (USD)", fontsize=11)
    ax.set_title("Price Movement During Order Lifetime", fontsize=12)
    ax.text(0.97, 0.97, "Positive = price rose\nNegative = price fell",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#555555")

    plt.tight_layout()
    out = out_dir / "fig1_lifecycle_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out.name}")


def fig_episode_timeline(
    df: pd.DataFrame,
    target_id: str,
    out_dir: Path,
) -> None:
    """
    Figure 2 – Spoofing Episode Timeline

    Zooms into the single largest spoofing episode and shows:
      · Top panel    : mid-price over a 60-second window centred on the episode
      · Bottom panel : individual order events by size, with spoof window shaded

    The three-stage spoofing pattern is immediately visible:
      [1] PLACE  — large limit orders appear on one side of the book
      [2] CANCEL — all layers pulled within seconds
      [3] PROFIT — small executable trade on the opposite side
    """
    target_ep = df[
        (df["trader_id"] == target_id) & (df["spoof_episode_id"] != "")
    ]
    if target_ep.empty:
        print("  No labeled episodes for target; skipping Fig 2.")
        return

    best_ep   = target_ep.groupby("spoof_episode_id")["quantity"].sum().idxmax()
    ep_events = df[df["spoof_episode_id"] == best_ep].sort_values("timestamp")

    t_start = ep_events["timestamp"].min() - pd.Timedelta(seconds=30)
    t_end   = ep_events["timestamp"].max() + pd.Timedelta(seconds=30)
    window  = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)].sort_values("timestamp")

    fig, (ax_p, ax_o) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(
        f"Spoofing Episode Timeline — Trader {target_id}  |  Episode {best_ep}",
        fontsize=13, fontweight="bold",
    )

    # ── Top: mid-price ─────────────────────────────────────────────────────────
    ax_p.plot(window["timestamp"], window["mid_price"],
              color="#2c3e50", linewidth=1.4, label="Mid-price")
    ax_p.set_ylabel("Mid-price (USD)", fontsize=11)
    ax_p.set_title("Market Mid-price with Spoofing Events", fontsize=12)
    ax_p.legend(fontsize=10)

    # ── Bottom: order events ───────────────────────────────────────────────────
    style_map = {
        "NEW_ORDER-BUY":     ("^", "#27ae60", 130),
        "NEW_ORDER-SELL":    ("^", "#c0392b", 130),
        "CANCEL-BUY":        ("v", "#a9cce3", 110),
        "CANCEL-SELL":       ("v", "#f1948a", 110),
        "FILL-BUY":          ("*", "#1a5276", 160),
        "FILL-SELL":         ("*", "#78281f", 160),
        "PARTIAL_FILL-BUY":  ("D", "#2e86c1", 100),
        "PARTIAL_FILL-SELL": ("D", "#922b21", 100),
    }
    plotted = set()
    for _, row in ep_events.iterrows():
        key    = f"{row['event_type']}-{row['side']}"
        marker, color, size = style_map.get(key, ("o", "gray", 80))
        label  = key.replace("-", " (") + ")" if key not in plotted else "_"
        plotted.add(key)
        ax_o.scatter(row["timestamp"], row["quantity"],
                     marker=marker, color=color, s=size, zorder=3, label=label)

    ax_o.set_ylabel("Quantity (contracts)", fontsize=11)
    ax_o.set_xlabel("Time (HH:MM:SS)", fontsize=11)
    ax_o.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_o.legend(fontsize=9, ncol=2, loc="upper right")

    # Highlight the spoof window
    t_new    = ep_events[ep_events["event_type"] == "NEW_ORDER"]["timestamp"].min()
    t_cancel = ep_events[ep_events["event_type"] == "CANCEL"]["timestamp"].max()
    for ax in (ax_p, ax_o):
        ax.axvspan(t_new, t_cancel, alpha=0.12, color="#e74c3c", zorder=0)
        ax.axvline(t_new,    color="#e74c3c", linestyle="--", linewidth=1.0)
        ax.axvline(t_cancel, color="#e74c3c", linestyle="--", linewidth=1.0)

    mid_t = t_new + (t_cancel - t_new) / 2
    y_top = window["mid_price"].max()
    ax_p.annotate("Spoof window", xy=(mid_t, y_top),
                  ha="center", va="top", fontsize=9,
                  color="#c0392b", fontweight="bold")

    plt.tight_layout()
    out = out_dir / "fig2_episode_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Metric Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _large_order_otr_per_window(
    orders_subset: pd.DataFrame,
    size_threshold: float,
    window_min: int = WINDOW_MIN,
) -> np.ndarray:
    """
    Compute large-order OTR (large-cancelled / filled) for each time window.

    large_order_otr = large_cancelled_orders / max(filled_orders, 1)

    This metric isolates the spoofing pattern more precisely than overall OTR:
    a legitimate passive trader with many unfilled small orders has a high
    standard OTR, but a very low large_order_otr.  A spoofer, by contrast,
    places many large orders that are systematically cancelled, producing a
    large_order_otr far above the market baseline.
    """
    sub = orders_subset.copy()
    session_start = sub["ts_open"].min().floor("h")
    sub["window"] = (
        (sub["ts_open"] - session_start).dt.total_seconds()
        // (window_min * 60)
    ).astype(int)

    values = []
    for _, g in sub.groupby("window"):
        large_cancelled = (g["is_cancelled"] & (g["quantity"] >= size_threshold)).sum()
        n_filled        = (~g["is_cancelled"]).sum()
        values.append(int(large_cancelled) / max(int(n_filled), 1))
    return np.array(values)


def fig_metric_comparison(
    orders: pd.DataFrame,
    target_id: str,
    out_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Figure 3 – Metric Comparison: Flagged Trader vs. Market

    Bar chart of three key metrics comparing the flagged trader's suspicious
    orders to all non-flagged market orders:
      · Median order duration (s)
      · Median order size (contracts)
      · Median large-order OTR (per 30-min window)

    Returns the underlying distribution arrays for Step 6 Mann-Whitney tests.
    """
    size_threshold = orders["quantity"].quantile(SIZE_QUANTILE)
    susp   = orders[(orders["trader_id"] == target_id) & orders["is_suspicious"]]
    market = orders[orders["trader_id"] != target_id]

    target_otr_arr = _large_order_otr_per_window(
        orders[orders["trader_id"] == target_id], size_threshold
    )
    market_otr_arr = _large_order_otr_per_window(market, size_threshold)

    metrics_display = {
        "Median Order\nDuration (s)": {
            f"Suspicious\n({target_id})": susp["order_duration_s"].median(),
            "Market\n(non-flagged)":      market["order_duration_s"].median(),
        },
        "Median Order\nSize (contracts)": {
            f"Suspicious\n({target_id})": susp["quantity"].median(),
            "Market\n(non-flagged)":      market["quantity"].median(),
        },
        "Large-Order OTR\n(per 30-min window)": {
            f"Suspicious\n({target_id})": float(np.median(target_otr_arr)),
            "Market\n(non-flagged)":       float(np.median(market_otr_arr)),
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Metric Comparison: Flagged Trader {target_id} (Suspicious Orders) vs. Market",
        fontsize=12, fontweight="bold",
    )

    for ax, (metric_name, vals) in zip(axes, metrics_display.items()):
        labels = list(vals.keys())
        values = list(vals.values())
        colors = [COLORS["suspicious"], COLORS["market"]]
        bars   = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.04,
                    f"{v:.2f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metric_name, fontsize=11)
        ax.set_ylim(0, max(values) * 1.35)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = out_dir / "fig3_metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out.name}")

    return {
        "order_duration_s": {
            "suspicious": susp["order_duration_s"].values,
            "market":     market["order_duration_s"].values,
        },
        "order_size": {
            "suspicious": susp["quantity"].values,
            "market":     market["quantity"].values,
        },
        "large_order_otr": {
            "suspicious": target_otr_arr,
            "market":     market_otr_arr,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 – Statistical Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(metric_arrays: dict) -> pd.DataFrame:
    """
    Mann-Whitney U tests on order duration, order size, and large-order OTR.

    H0 : The distribution of the flagged trader's suspicious orders does not
         differ from the distribution of non-flagged market orders.

    H1 : Suspicious flagged-trader orders show (one-sided, directional):
           · shorter order duration    (alternative = 'less')
           · larger order size         (alternative = 'greater')
           · higher large-order OTR    (alternative = 'greater')

    Why Mann-Whitney U:
      · Order duration, size, and OTR are heavily right-skewed — parametric
        tests (t-test) are inappropriate.
      · The test compares medians via ranks without distributional assumptions.
      · Both groups have large samples, validating the normal approximation of U.

    This mirrors the non-parametric approach used in CFTC-adjacent economic
    analysis, where distributional assumptions cannot be imposed on
    market-microstructure data.
    """
    sep = "─" * 70
    print(f"\n{sep}")
    print("  STATISTICAL TESTS  (Mann-Whitney U, alpha = 0.05)")
    print(sep)

    tests = [
        (
            "Order Duration (s)",
            "order_duration_s",
            "less",
            "suspicious orders placed and cancelled faster than typical market orders",
        ),
        (
            "Order Size (contracts)",
            "order_size",
            "greater",
            "suspicious orders are disproportionately large relative to the market",
        ),
        (
            "Large-Order OTR (per window)",
            "large_order_otr",
            "greater",
            "flagged trader places far more large cancelled orders per executed trade",
        ),
    ]

    results = []
    for display_name, key, alternative, rationale in tests:
        x = metric_arrays[key]["suspicious"]
        y = metric_arrays[key]["market"]

        if len(x) < 2 or len(y) < 2:
            print(f"\n  {display_name}: insufficient data, skipping.")
            continue

        stat, p = mannwhitneyu(x, y, alternative=alternative, use_continuity=True)
        reject  = bool(p < ALPHA)

        print(f"\n  Metric           : {display_name}")
        print(f"  Rationale        : {rationale}")
        print(f"  H1 direction     : suspicious '{alternative}' than market")
        print(f"  n (suspicious)   : {len(x):,}    n (market) : {len(y):,}")
        print(f"  Median suspicious: {np.median(x):.4f}")
        print(f"  Median market    : {np.median(y):.4f}")
        print(f"  U-statistic      : {stat:,.0f}")
        print(f"  p-value          : {p:.4e}")
        print(f"  Decision         : {'REJECT H0' if reject else 'FAIL TO REJECT H0'}"
              f"  (alpha = {ALPHA})")

        results.append({
            "Metric":                  display_name,
            "n_suspicious":            len(x),
            "n_market":                len(y),
            "Median_suspicious":       round(float(np.median(x)), 4),
            "Median_market":           round(float(np.median(y)), 4),
            "U_statistic":             round(float(stat), 1),
            "p_value":                 round(float(p), 8),
            "Reject_H0 (alpha=0.05)":  reject,
        })

    print(f"\n{sep}")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  Spoofing Detection Pipeline  |  ESZ25")
    print(f"{sep}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[Step 1] ETL — loading and reconstructing orders ...")
    df     = load_events(DATA_PATH)
    orders = reconstruct_orders(df)

    print("\n[Step 2] Computing CFTC-based spoofing metrics ...")
    orders, trader_stats = compute_features(orders)
    trader_stats.to_csv(OUTPUT_DIR / "trader_metrics.csv", index=False)
    print(f"  Saved trader_metrics.csv  ({len(trader_stats)} traders)")

    print("\n[Step 3] Identifying flagged trader ...")
    target_id = identify_flagged_trader(trader_stats)
    susp_orders = orders[
        (orders["trader_id"] == target_id) & orders["is_suspicious"]
    ]
    susp_orders.to_csv(OUTPUT_DIR / "suspicious_orders.csv", index=False)
    print(f"  {len(susp_orders):,} suspicious orders -> suspicious_orders.csv")

    print("\n[Step 4] Generating lifecycle visualisations ...")
    fig_lifecycle_comparison(orders, target_id, OUTPUT_DIR)
    fig_episode_timeline(df, target_id, OUTPUT_DIR)

    print("\n[Step 5] Comparing metrics: flagged trader vs. market ...")
    metric_arrays = fig_metric_comparison(orders, target_id, OUTPUT_DIR)

    print("\n[Step 6] Running statistical inference ...")
    results = run_statistical_tests(metric_arrays)
    results.to_csv(OUTPUT_DIR / "statistical_test_results.csv", index=False)
    print(f"\n  Test results -> statistical_test_results.csv")

    print(f"\n{sep}")
    print(f"  Pipeline complete.  All outputs -> {OUTPUT_DIR}/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
