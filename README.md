[README.md](https://github.com/user-attachments/files/28443273/README.md)
# Spoofing-Detection
Synthetic dataset and spoofing detection code.
# Spoofing Detection Pipeline — ESZ25
**Natasha Shen** | [natashashen04@gmail.com](mailto:natashashen04@gmail.com) | [linkedin.com/in/natasha-shen](https://www.linkedin.com/in/natasha-shen)

---

## Project Overview

This repository presents a Python pipeline for detecting order-book spoofing in
financial futures markets. The methodology is modeled on the analytical framework
I applied on a financial market manipulation
case, and mirrors the four-step approach used in CFTC enforcement proceedings.

The project demonstrates end-to-end data skills — synthetic data generation,
ETL, feature engineering, data visualization, and non-parametric statistical
inference — applied to a domain directly relevant to the OAG's Economic Justice
and Investor Protection work.

---

## Background

**Spoofing** is the practice of placing large limit orders with no intention of
executing them, creating a false impression of supply or demand to move prices
before cancelling the orders and trading profitably at the manipulated price.
It has been a federal crime under the Dodd-Frank Act (7 U.S.C. § 6c(a)(5)(C))
since 2010, and has been the subject of major enforcement actions by the CFTC
and DOJ — including cases involving E-mini S&P 500 futures, the exact instrument
simulated here.

The OAG's mandate to protect New York investors and investigate financial fraud
makes this type of quantitative market-surveillance methodology directly
applicable to the role.

---

## Repository Structure

```
.
├── generate_market_data.py       # Step 0: Synthetic LOB data generator
├── spoof_detection.py            # Steps 1–6: Detection pipeline
├── README.md                     # This file
│
└── spoof_detection_report/       # Auto-generated outputs
    ├── trader_metrics.csv
    ├── suspicious_orders.csv
    ├── statistical_test_results.csv
    ├── fig1_lifecycle_comparison.png
    ├── fig2_episode_timeline.png
    └── fig3_metric_comparison.png
```

---

## Data

### Why synthetic data?

Real limit-order-book datasets (e.g., LOBSTER, Nasdaq TotalView-ITCH) do not
carry ground-truth spoofing labels. Using a synthetic dataset allows the
detection algorithm to be **evaluated against known outcomes** — a rigorous
approach consistent with reproducible research standards. The generator is
calibrated to match the statistical properties of real ES futures order flow:
millisecond-precision timestamps, realistic price dynamics, and Poisson-distributed
order arrivals.

### What the dataset contains

`generate_market_data.py` simulates one full trading session
(09:30–16:00 ET, 15 December 2025) of the **E-mini S&P 500 futures contract
(ESZ25)** — the instrument at the center of landmark CFTC spoofing prosecutions.

| Field | Description |
|---|---|
| `event_id` | Monotonically increasing event counter |
| `timestamp` | Millisecond-precision event time |
| `order_id` | Unique order identifier |
| `trader_id` | Anonymized participant (`T###` = legitimate, `S###` = spoofer) |
| `event_type` | `NEW_ORDER` / `CANCEL` / `FILL` / `PARTIAL_FILL` |
| `side` | `BUY` / `SELL` |
| `price` | Limit price (multiple of tick size = $0.25) |
| `quantity` | Order size in contracts |
| `best_bid`, `best_ask`, `mid_price` | Market state at event time |
| `is_spoof` | Ground-truth label for injected spoofing episodes |
| `spoof_episode_id` | Groups all events belonging to one episode |

**Simulation output:** 170,123 events across 50 traders (45 legitimate, 5
spoofers), including 194 injected spoofing episodes.

---

## Methodology

### Step 1 — ETL

Raw order-book events are loaded and reconstructed into **per-order lifecycle
records** by joining each `NEW_ORDER` event to its terminal event (`CANCEL`,
`FILL`, or `PARTIAL_FILL`). Derived fields include order duration, cancellation
flag, and mid-price change during the order's life.

### Step 2 — Feature Engineering

Three CFTC-based metrics are computed for each order and aggregated per trader:

| Metric | Definition | Spoofing Signal |
|---|---|---|
| `order_duration_s` | Seconds from placement to cancellation/fill | Spoofers cancel in < 5 s |
| `order size` | Contracts per order | Spoof orders are disproportionately large |
| `large_order_otr` | Large-cancelled orders / filled orders per 30-min window | Spoofers inflate this ratio far above market norms |

The order-level **suspicious flag** requires all three conditions to hold:
order cancelled + duration < 5 s + size in the top quartile of all orders.

### Step 3 — Flagging

The trader with the highest `suspicious_rate` (proportion of orders meeting all
three criteria) is identified as the flagged trader. In actual CFTC enforcement,
the regulator identifies this trader through market surveillance and whistleblower
referrals before economic analysis begins — a starting point this pipeline
mirrors for demonstration purposes.

**Flagged trader (S001):**
- Suspicious order rate: **71.4%**
- Median order duration: **2.25 seconds**
- Median order size: **116 contracts** (vs. 8 for the market)

### Step 4 — Lifecycle Comparison

Two visualizations contrast the flagged trader's suspicious and non-suspicious
orders:

**Figure 1 — Order Lifecycle Comparison**

The duration distribution shows suspicious orders clustering below the 5-second
threshold — an immediate and sharp departure from legitimate passive trading.
The price-impact panel shows that suspicious orders coincide with larger
mid-price movements, consistent with manufactured order-book pressure.

![Fig 1](spoof_detection_report/fig1_lifecycle_comparison.png)

**Figure 2 — Episode Timeline**

A single spoofing episode, zoomed to a 60-second window, illustrates the
three-stage pattern:

1. **Place** — large limit orders appear on one side of the book
2. **Cancel** — all layers pulled within seconds (shaded window)
3. **Profit** — small trade executed on the opposite side at the manipulated price

![Fig 2](spoof_detection_report/fig2_episode_timeline.png)

### Step 5 — Metric Comparison

Median values for each metric are compared between the flagged trader's
suspicious orders and all non-flagged market orders.

![Fig 3](spoof_detection_report/fig3_metric_comparison.png)

### Step 6 — Statistical Inference

**Mann-Whitney U tests** are applied to each of the three metrics. The
Mann-Whitney test is appropriate here because order duration, size, and OTR
distributions are heavily right-skewed, making parametric tests (t-test, ANOVA)
inappropriate. The test compares medians via ranks without distributional
assumptions and is valid for the large sample sizes involved.

**Null hypothesis (H₀):** The distribution of the flagged trader's suspicious
orders does not differ from that of non-flagged market orders.

**Alternative hypothesis (H₁):** Suspicious orders are shorter, larger, and
associated with a higher large-order OTR (one-sided, directional).

| Metric | Median (Suspicious) | Median (Market) | p-value | Decision |
|---|---|---|---|---|
| Order Duration (s) | 3.29 | 27.80 | 4.20 × 10⁻¹⁴ | Reject H₀ |
| Order Size (contracts) | 152.0 | 8.0 | 7.86 × 10⁻⁶⁴ | Reject H₀ |
| Large-Order OTR (per window) | 2.50 | 0.70 | 5.45 × 10⁻⁰⁶ | Reject H₀ |

All three differences are statistically significant at the 5% level. The
effect sizes are large — median suspicious order duration is **8.5×** shorter
and median order size is **19×** larger than the market baseline — consistent
with the deliberate, systematic nature of spoofing behavior.

---

## Setup & Usage

### Requirements

```
Python >= 3.10
pandas
numpy
scipy
matplotlib
seaborn
```

Install dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn
```

### Run

```bash
# Step 1: Generate the synthetic dataset
python generate_market_data.py
# → market_data_ESZ25.csv

# Step 2: Run the detection pipeline
python spoof_detection.py
# → spoof_detection_report/
```

Both scripts are fully reproducible: `generate_market_data.py` uses a fixed
random seed (`SEED = 42`) and `spoof_detection.py` produces deterministic
outputs given the same input data.

---

## Notes & Limitations

**Metric calibration.** In documented CFTC spoofing cases (e.g., *CFTC v. Sarao*),
order-to-trade ratios frequently reach 100:1 or higher, and cancellation windows
are often sub-second. The thresholds used here (`DURATION_THRESHOLD_S = 5.0`,
`SUSPICIOUS_RATE_FLAG = 0.50`) are calibrated to the simulation's parameters,
where legitimate passive traders also have elevated cancellation rates due to
resting-order dynamics. In production, thresholds would be set empirically from
regulatory guidance and the specific market's baseline statistics.

**Ground-truth labels.** The `is_spoof` column exists only because the dataset
is synthetic. In a real investigation, ground truth is established through legal
proceedings, not pre-labeled data. The pipeline's detection logic (`is_suspicious`)
operates independently of this column, using only observable market signals.

**Single instrument, single session.** The pipeline is designed for extension
to multi-day, multi-instrument settings — a natural next step for production use.

---

*This code sample was prepared as part of an application for the Data Analyst
position at the New York Office of the Attorney General (Ref: RAD_NYC_DAT_6444).*
