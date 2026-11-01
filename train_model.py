"""
RxLedger - ML demand forecasting pipeline (course end project)

Steps: 1) build a 2-year daily pharmacy sales dataset  2) feature engineering
(lags, rolling means, calendar, weather, festival, promo)  3) time-based
train/test split  4) train & compare models  5) recursive 14-day forecast
6) export results (CSV, JSON, PNG) used by the RxLedger web app.

Run:  pip install numpy pandas scikit-learn matplotlib
      python train_model.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
DAYS, HORIZON, TEST_DAYS = 730, 14, 90

# name: (category, base units/day, seasonal amplitude, peak day-of-year, 2-year growth)
MEDS = {
    "Paracetamol 650mg":  ("Pain & Fever", 9, .25, 15, .05),
    "Amoxicillin 500mg":  ("Antibiotic", 4, .30, 15, .05),
    "Metformin 500mg":    ("Diabetes", 8, .03, 1, .06),
    "Azithromycin 500mg": ("Antibiotic", 3, .20, 15, .70),
    "Pantoprazole 40mg":  ("Gastro", 6, .05, 1, .05),
    "Cetirizine 10mg":    ("Allergy", 7, .35, 90, .04),
    "Amlodipine 5mg":     ("Cardiac", 5, .03, 1, .05),
    "Vitamin D3 60K":     ("Vitamins", 2, .05, 1, .60),
    "Cough Syrup 100ml":  ("Respiratory", 1.2, .50, 10, .03),
    "Ibuprofen 400mg":    ("Pain & Fever", 5, .10, 15, .04),
    "Insulin Glargine":   ("Diabetes", .7, .02, 1, .06),
    "ORS Sachet":         ("Other", 3, .55, 140, .50),
}
NAMES = list(MEDS)

# ---------------------------------------------------------------- 1. dataset
end = pd.Timestamp.today().normalize()
dates = pd.date_range(end - pd.Timedelta(days=DAYS - 1), end)
n = len(dates)
doy, dow = dates.dayofyear.values, dates.dayofweek.values
t = np.arange(n) / n
festival = np.zeros(n, dtype=int)
festival[rng.choice(n, 28, replace=False)] = 1
temp = 28 + 7 * np.sin(2 * np.pi * (doy - 49) / 365) + rng.normal(0, 1.5, n)
weekly = np.array([1.05, 1.0, 1.0, 1.0, 1.05, 1.12, .8])   # Mon..Sun

rows = []
for name, (cat, base, amp, peak, growth) in MEDS.items():
    promo = (rng.random(n) < .08).astype(int)
    lam = (base * (1 + amp * np.cos(2 * np.pi * (doy - peak) / 365))
           * (1 + growth * t * 2) * weekly[dow] * (1 + .2 * festival) * (1 + .3 * promo))
    if name == "ORS Sachet":
        lam = lam * (1 + .04 * (temp - 28))
    units = rng.poisson(np.clip(lam, .05, None))
    rows.append(pd.DataFrame({"date": dates, "medicine": name, "category": cat,
                              "units_sold": units, "temperature_c": temp.round(1),
                              "is_festival": festival, "promo": promo}))
data = pd.concat(rows, ignore_index=True)
data.to_csv("pharma_sales.csv", index=False)
print(f"Dataset: {len(data):,} rows, {len(NAMES)} medicines, {n} days")

# ------------------------------------------------------ 2. feature engineering
def row_features(name, d, temp_c, fest, promo, hist):
    """Features for one medicine on one day, given its past sales `hist`."""
    f = {"dow": d.dayofweek, "month": d.month, "temp": temp_c, "festival": fest, "promo": promo,
         "lag_1": hist[-1], "lag_7": hist[-7], "lag_14": hist[-14],
         "roll_7": np.mean(hist[-7:]), "roll_28": np.mean(hist[-28:])}
    for m in NAMES:
        f["med_" + m] = int(m == name)
    return f

recs, ys, keys = [], [], []
for name, g in data.groupby("medicine"):
    g = g.sort_values("date").reset_index(drop=True)
    for i in range(28, len(g)):
        r = g.iloc[i]
        recs.append(row_features(name, r["date"], r["temperature_c"], r["is_festival"],
                                 r["promo"], g["units_sold"].values[:i]))
        ys.append(r["units_sold"]); keys.append((name, r["date"]))
X = pd.DataFrame(recs)
y = np.array(ys, dtype=float)
meta = pd.DataFrame(keys, columns=["medicine", "date"])

# ------------------------------------------------------- 3. time-based split
split = dates[-TEST_DAYS]
tr, te = (meta["date"] < split).values, (meta["date"] >= split).values
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
print(f"Train rows: {tr.sum():,}   Test rows: {te.sum():,}   Split date: {split.date()}")

# ------------------------------------------------------------ 4. train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=.05, max_depth=3, random_state=42),
}

def evaluate(name, pred):
    return {"name": name,
            "mae": round(float(mean_absolute_error(yte, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 3),
            "r2": round(float(r2_score(yte, pred)), 3),
            "wape": round(float(np.abs(yte - pred).sum() / yte.sum() * 100), 2)}

results = [evaluate("Baseline (7-day average)", Xte["roll_7"].values)]
fitted = {}
for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    fitted[name] = mdl
    results.append(evaluate(name, mdl.predict(Xte)))
for r in results:
    print(f"{r['name']:<26} MAE {r['mae']:.3f}  RMSE {r['rmse']:.3f}  R2 {r['r2']:.3f}  WAPE {r['wape']}%")

best = min(results[1:], key=lambda r: r["rmse"])["name"]
print("Best model:", best)

# feature importance (Random Forest), medicine one-hots merged
imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns)
imp_m = imp[[c for c in X.columns if c.startswith("med_")]].sum()
imp = imp[[c for c in X.columns if not c.startswith("med_")]]
imp["medicine"] = imp_m
imp = imp.sort_values(ascending=False)

# actual vs predicted (test period, all medicines summed per day)
pred_best = fitted[best].predict(Xte)
cmp = meta[te].assign(actual=yte, pred=pred_best).groupby("date")[["actual", "pred"]].sum().tail(45)

# ------------------------------------------- 5. retrain on all data + forecast
final = clone(models[best]).fit(X, y)
hist = {m: list(data[data.medicine == m].sort_values("date")["units_sold"].values.astype(float)) for m in NAMES}
fc = {m: [] for m in NAMES}
for h in range(1, HORIZON + 1):
    d = end + pd.Timedelta(days=h)
    exp_temp = 28 + 7 * np.sin(2 * np.pi * (d.dayofyear - 49) / 365)
    batch = pd.DataFrame([row_features(m, d, exp_temp, 0, 0, hist[m]) for m in NAMES])[X.columns]
    for m, p in zip(NAMES, final.predict(batch)):
        p = max(0.0, float(p)); fc[m].append(round(p, 1)); hist[m].append(p)

# ---------------------------------------------------------------- 6. export
out = {"generated": str(end.date()), "best": best,
       "dataset": {"rows": int(len(data)), "medicines": len(NAMES), "days": n,
                   "train": int(tr.sum()), "test": int(te.sum()), "split": str(split.date())},
       "models": results,
       "importance": [{"f": k, "v": round(float(v), 3)} for k, v in imp.head(7).items()],
       "curve": {"actual": cmp["actual"].round(0).tolist(), "pred": cmp["pred"].round(1).tolist()},
       "meds": {m: {"actual30": [int(v) for v in data[data.medicine == m].sort_values("date")["units_sold"].tail(30)],
                    "forecast14": fc[m]} for m in NAMES}}
json.dump(out, open("ml_results.json", "w"), separators=(",", ":"))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].bar([r["name"].split(" (")[0] for r in results], [r["rmse"] for r in results], color="#0b7a63")
ax[0].set_title("Model comparison (RMSE, lower is better)"); ax[0].tick_params(axis="x", rotation=20)
ax[1].plot(cmp["actual"].values, label="Actual", color="#0a5c4b"); ax[1].plot(cmp["pred"].values, label=best, color="#6b4fbb")
ax[1].set_title("Actual vs predicted (test period)"); ax[1].legend()
ax[2].barh(imp.index[::-1], imp.values[::-1], color="#0b7a63"); ax[2].set_title("Feature importance (Random Forest)")
plt.tight_layout(); plt.savefig("model_report.png", dpi=140)
print("Saved pharma_sales.csv, ml_results.json, model_report.png")
