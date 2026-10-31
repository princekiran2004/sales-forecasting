"""
Sales Forecasting: trend, seasonality and a model comparison
Topics: data type classification | derived metrics | group by | trend analysis | correlation (+ forecasting)
Stack: Python, pandas, NumPy, Matplotlib, Seaborn, statsmodels
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
IMG = "images/"; os.makedirs(IMG, exist_ok=True); os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(13)
NAVY, TEAL, ORG, GREY = "#264653", "#2A9D8F", "#E76F51", "#B0B8C4"

# ------------------------------------------------------------------
# 0. DATA (synthetic daily sales; replace with pd.read_csv for real data)
#    Required columns: date, category, region, orders, sales, marketing_spend
# ------------------------------------------------------------------
days = pd.date_range("2022-10-01", "2026-09-30")
month_idx = {1: .82, 2: .85, 3: .95, 4: .97, 5: 1.0, 6: .96, 7: .93, 8: .98, 9: 1.04, 10: 1.12, 11: 1.32, 12: 1.45}
cats = {"Electronics": 1.3, "Clothing": 1.0, "Home": .8, "Beauty": .5}
regs = {"North": 1.2, "South": .9, "West": 1.0}
rows = []
t = np.arange(len(days))
mkey = days.year * 12 + days.month                                      # month-level shocks shared by all series (market-wide ups and downs)
shock_map = dict(zip(np.unique(mkey), rng.lognormal(0, .07, len(np.unique(mkey)))))
shock_d = np.array([shock_map[k] for k in mkey])
for cat, cw in cats.items():
    for reg, rw in regs.items():
        base = 520 * cw * rw
        trend = 1 + 0.12 * t / 365                                          # about 12% growth per year
        season = np.array([month_idx[d.month] for d in days])
        promo = rng.random(len(days)) < 0.06
        weekday = np.where(days.dayofweek >= 5, 1.12, 1.0)
        sales = base * trend * season * weekday * (1 + 0.18 * promo) * shock_d * rng.lognormal(0, .12, len(days))
        orders = np.maximum(1, (sales / rng.uniform(38, 46, len(days))).astype(int))
        mkt = sales * rng.uniform(.07, .11, len(days)) * (1 + .4 * promo)
        rows.append(pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "category": cat, "region": reg, "orders": orders,
                                  "sales": sales.round(2), "marketing_spend": mkt.round(2), "promo_day": promo.astype(int)}))
raw = pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------------
# 1. DATA TYPE CLASSIFICATION + QUALITY CHECK
# ------------------------------------------------------------------
def classify(col, s):
    if col == "date": return "Datetime (stored as text)"
    if col == "promo_day": return "Binary"
    if pd.api.types.is_numeric_dtype(s): return "Numeric - Discrete" if col == "orders" else "Numeric - Continuous"
    return "Categorical"
types = pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values, "classification": [classify(c, raw[c]) for c in raw.columns],
                      "missing": raw.isna().sum().values, "unique_values": [raw[c].nunique() for c in raw.columns]})
print("=== 1. DATA TYPE CLASSIFICATION ===\n", types.to_string(index=False))
df = raw.assign(date=pd.to_datetime(raw["date"]))
full = pd.date_range(df.date.min(), df.date.max()); missing_days = len(full.difference(df.date.unique()))
print(f"\nRows: {len(df):,} | Date range: {df.date.min().date()} to {df.date.max().date()} | Missing calendar days: {missing_days}")
print(f"Duplicate (date, category, region) rows: {df.duplicated(['date', 'category', 'region']).sum()}")

fig, ax = plt.subplots(figsize=(8.5, 4.5))
cnt = types["classification"].value_counts(); b = ax.barh(cnt.index, cnt.values, color=sns.color_palette("deep", len(cnt)))
ax.bar_label(b, padding=3); ax.set_title("Columns by Data Type Classification", fontweight="bold"); ax.set_xlabel("Number of columns")
plt.tight_layout(); plt.savefig(IMG + "1_data_types.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 2. DERIVED METRICS + MONTHLY SERIES
# ------------------------------------------------------------------
m = df.set_index("date").resample("MS").agg(sales=("sales", "sum"), orders=("orders", "sum"), marketing_spend=("marketing_spend", "sum"), promo_days=("promo_day", "sum"))
m["avg_order_value"] = m["sales"] / m["orders"]
m["mom_growth_%"] = m["sales"].pct_change() * 100
m["yoy_growth_%"] = m["sales"].pct_change(12) * 100
m["roll_3m"] = m["sales"].rolling(3).mean(); m["roll_12m"] = m["sales"].rolling(12).mean()
m["mkt_to_sales_%"] = m["marketing_spend"] / m["sales"] * 100
m["month"] = m.index.month; m["year"] = m.index.year
print("\n=== 2. MONTHLY SERIES (last 4) ===\n", m[["sales", "orders", "avg_order_value", "mom_growth_%", "yoy_growth_%"]].tail(4).round(1))

# ------------------------------------------------------------------
# 3. GROUP BY
# ------------------------------------------------------------------
by_cat = df.groupby("category")["sales"].sum().sort_values(ascending=False); by_reg = df.groupby("region")["sales"].sum().sort_values(ascending=False)
yr = df.assign(fy=np.where(df.date.dt.month <= 9, df.date.dt.year, df.date.dt.year + 1)).groupby("fy")["sales"].sum()   # 12-month years ending September
seas = m.groupby("month")["sales"].mean(); seas_idx = seas / seas.mean()
print("\n=== 3. GROUP BY ===\nSales by category (%):\n", (by_cat / by_cat.sum() * 100).round(1), "\nSeasonal index (peak/low):",
      seas_idx.idxmax(), round(seas_idx.max(), 2), "/", seas_idx.idxmin(), round(seas_idx.min(), 2))

fig, axes = plt.subplots(1, 3, figsize=(17, 5), width_ratios=[1, 1, 1.4])
axes[0].barh(by_cat.index[::-1], by_cat.values[::-1] / 1e6, color=TEAL); axes[0].set_title("Sales by Category (millions)", fontweight="bold")
axes[1].barh(by_reg.index[::-1], by_reg.values[::-1] / 1e6, color=NAVY); axes[1].set_title("Sales by Region (millions)", fontweight="bold")
cols = [ORG if v >= 1.1 else (TEAL if v >= 1 else GREY) for v in seas_idx.values]
axes[2].bar(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], seas_idx.values, color=cols)
axes[2].axhline(1, color="black", ls="--", lw=1); axes[2].set_title("Seasonal Index by Month (1.0 = average month)", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG + "2_group_by_seasonality.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 4. TREND ANALYSIS: decomposition and growth
# ------------------------------------------------------------------
dec = seasonal_decompose(m["sales"], model="multiplicative", period=12)
trend_growth = (dec.trend.dropna().iloc[-1] / dec.trend.dropna().iloc[0]) ** (12 / (len(dec.trend.dropna()) - 1)) - 1
print(f"\n=== 4. TREND ===\nAnnualised trend growth: {trend_growth * 100:.1f}% | Latest YoY: {m['yoy_growth_%'].iloc[-1]:.1f}% | 12-month sales ending Sep (millions):\n{(yr / 1e6).round(2)}")
fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for a, (s, title, c) in zip(ax, [(m["sales"], "Observed monthly sales", NAVY), (dec.trend, "Trend", ORG), (dec.seasonal, "Seasonal factor", TEAL), (dec.resid, "Residual", GREY)]):
    a.plot(s.index, s.values, color=c, lw=2); a.set_title(title, fontweight="bold", loc="left", fontsize=11)
ax[2].axhline(1, color="black", lw=.8, ls="--"); ax[3].axhline(1, color="black", lw=.8, ls="--")
plt.tight_layout(); plt.savefig(IMG + "3_decomposition.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
ax[0].plot(m.index, m["sales"], color=GREY, label="Monthly sales"); ax[0].plot(m.index, m["roll_12m"], color=NAVY, lw=3, label="12-month rolling average"); ax[0].legend()
ax[0].set_title("Sales with 12-Month Rolling Average", fontweight="bold")
y = m["yoy_growth_%"].dropna(); ax[1].bar(y.index, y.values, width=20, color=np.where(y.values >= 0, TEAL, ORG)); ax[1].axhline(y.mean(), color="black", ls="--", label=f"mean {y.mean():.1f}%")
ax[1].legend(); ax[1].set_title("Year-over-Year Growth %", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG + "4_trend_growth.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 5. CORRELATION + AUTOCORRELATION
# ------------------------------------------------------------------
m["time_index"] = np.arange(len(m)); m["seasonal_factor"] = dec.seasonal.values
corr = m[["sales", "orders", "marketing_spend", "promo_days", "time_index", "seasonal_factor"]].corr()
ac = acf(m["sales"], nlags=14)
print("\n=== 5. CORRELATION ===\n", corr["sales"].round(2), "\nACF lag 1 / lag 12:", round(ac[1], 2), "/", round(ac[12], 2))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), width_ratios=[1.2, 1])
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), 1), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title("Correlation of Monthly Metrics", fontweight="bold")
ax[1].bar(range(len(ac)), ac, color=[ORG if i == 12 else NAVY for i in range(len(ac))]); ax[1].axhline(0, color="black", lw=.8)
ax[1].set_title("Autocorrelation of Monthly Sales (lag in months)", fontweight="bold"); ax[1].set_xlabel("Lag")
plt.tight_layout(); plt.savefig(IMG + "5_correlation_autocorrelation.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 6. FORECASTING: compare models on the last 12 months
# ------------------------------------------------------------------
y = m["sales"]; H = 12; train, test = y.iloc[:-H], y.iloc[-H:]
def metrics(a, f): e = np.asarray(a) - np.asarray(f); return pd.Series({"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAPE_%": (np.abs(e) / np.asarray(a)).mean() * 100})
def lin_seasonal(tr, h):
    n = len(tr); X = lambda idx: np.column_stack([np.ones(len(idx)), idx] + [(idx % 12 == k).astype(float) for k in range(1, 12)])
    beta = np.linalg.lstsq(X(np.arange(n)), np.log(tr.values), rcond=None)[0]; return np.exp(X(np.arange(n, n + h)) @ beta)
def hw(tr, h): return ExponentialSmoothing(tr, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(h).values
models = {"Naive (last value)": lambda tr, h: np.repeat(tr.iloc[-1], h),
          "3-month moving average": lambda tr, h: np.repeat(tr.iloc[-3:].mean(), h),
          "Seasonal naive (same month last year)": lambda tr, h: tr.iloc[-12:].values[:h],
          "Trend + seasonality (regression)": lin_seasonal, "Holt-Winters": hw}
preds = {k: f(train, H) for k, f in models.items()}
res = pd.DataFrame({k: metrics(test.values, v) for k, v in preds.items()}).T.sort_values("MAPE_%")
print("\n=== 6. MODEL COMPARISON (test = last 12 months) ===\n", res.round(1))
best = res.index[0]

# refit best on all data and forecast the next 6 months with an approximate 95% interval
FH = 6; fut_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=FH, freq="MS")
fc = models[best](y, FH); rmse = res.loc[best, "RMSE"]
future = pd.DataFrame({"forecast": fc, "lower_95": fc - 1.96 * rmse, "upper_95": fc + 1.96 * rmse}, index=fut_idx)
print(f"\nBest model: {best} (MAPE {res.loc[best, 'MAPE_%']:.1f}%)\nNext {FH} months forecast:\n", future.round(0))

fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), width_ratios=[1, 1.5])
r = res["MAPE_%"].iloc[::-1]; b = ax[0].barh(r.index, r.values, color=[TEAL if i == best else GREY for i in r.index]); ax[0].bar_label(b, fmt="%.1f%%", padding=3)
ax[0].set_title("Model Comparison: Test Error (MAPE, lower is better)", fontweight="bold")
ax[1].plot(y.index, y.values, color=NAVY, lw=2, label="Actual"); ax[1].plot(test.index, preds[best], color=ORG, lw=2, ls="--", label=f"{best} (test)")
ax[1].plot(future.index, future.forecast, color=TEAL, lw=3, marker="o", label="Forecast"); ax[1].fill_between(future.index, future.lower_95, future.upper_95, color=TEAL, alpha=.2, label="~95% range")
ax[1].axvline(test.index[0], color="gray", ls=":"); ax[1].set_title("Forecast: Next 6 Months", fontweight="bold"); ax[1].legend()
ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout(); plt.savefig(IMG + "6_forecast.png", dpi=200); plt.close()

# ------------------------------------------------------------------
# 7. LINKEDIN DASHBOARD
# ------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9)); fig.suptitle("Sales Forecasting  |  Trend, Seasonality & Model Comparison", fontsize=20, fontweight="bold", y=.98)
gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.25)
a1 = fig.add_subplot(gs[0, 0]); a1.plot(m.index, m["sales"] / 1e3, color=GREY); a1.plot(m.index, m["roll_12m"] / 1e3, color=NAVY, lw=3); a1.set_title("Trend: Monthly Sales + 12M Average (thousands)", fontweight="bold")
for _a in (a1,): _a.xaxis.set_major_locator(mdates.MonthLocator(interval=12)); _a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
a2 = fig.add_subplot(gs[0, 1]); a2.bar(range(12), seas_idx.values, color=cols); a2.set_xticks(range(12)); a2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8); a2.axhline(1, color="black", ls="--", lw=1)
a2.set_title("Group By: Seasonal Index by Month", fontweight="bold")
a3 = fig.add_subplot(gs[1, 0]); b = a3.barh(r.index, r.values, color=[TEAL if i == best else GREY for i in r.index]); a3.bar_label(b, fmt="%.1f%%", padding=3); a3.set_xlim(0, r.max() * 1.25)
a3.set_title("Model Error on Last 12 Months (MAPE)", fontweight="bold")
a4 = fig.add_subplot(gs[1, 1]); a4.plot(y.index[-24:], y.values[-24:] / 1e3, color=NAVY, lw=2, label="Actual"); a4.plot(future.index, future.forecast / 1e3, color=TEAL, lw=3, marker="o", label="Forecast")
a4.fill_between(future.index, future.lower_95 / 1e3, future.upper_95 / 1e3, color=TEAL, alpha=.2); a4.legend(); a4.set_title("Forecast: Next 6 Months (thousands)", fontweight="bold")
a4.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); a4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.savefig(IMG + "0_linkedin_dashboard.png", dpi=200, bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------
# INSIGHTS + EXPORT
# ------------------------------------------------------------------
naive_s = res.loc["Seasonal naive (same month last year)", "MAPE_%"]
print("\n=== KEY INSIGHTS ===")
print(f"- Total sales {df.sales.sum() / 1e6:.1f}M | top category {by_cat.index[0]} ({by_cat.iloc[0] / by_cat.sum() * 100:.1f}%) | top region {by_reg.index[0]}")
print(f"- Seasonality: peak {seas_idx.idxmax()} (index {seas_idx.max():.2f}), low {seas_idx.idxmin()} ({seas_idx.min():.2f}); Nov+Dec = {(seas.loc[[11, 12]].sum() / seas.sum() * 100):.1f}% of the year")
print(f"- Trend growth ~{trend_growth * 100:.1f}% a year; latest YoY {m['yoy_growth_%'].iloc[-1]:.1f}%")
print(f"- Best model: {best} MAPE {res.loc[best, 'MAPE_%']:.1f}% vs naive {res.loc['Naive (last value)', 'MAPE_%']:.1f}% and seasonal naive {naive_s:.1f}%")
print(f"- Corr sales vs time {corr.loc['sales', 'time_index']:.2f} | vs seasonal factor {corr.loc['sales', 'seasonal_factor']:.2f} | vs promo days {corr.loc['sales', 'promo_days']:.2f} | vs marketing spend {corr.loc['sales', 'marketing_spend']:.2f} (mechanical) | ACF lag 12 {ac[12]:.2f}")
m.round(2).to_csv("data/monthly_sales.csv"); future.round(0).to_csv("data/forecast_next_6_months.csv"); res.round(2).to_csv("data/model_comparison.csv")
print("\nSaved data/monthly_sales.csv, data/forecast_next_6_months.csv, data/model_comparison.csv")
