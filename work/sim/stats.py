"""Статистична обробка результатів: описові характеристики, ДІ Стьюдента, тести
нормальності (Шапіро-Вілка, хі-квадрат), ідентифікація закону розподілу, регресія."""
import numpy as np
from scipy import stats as st


def describe(x, conf=0.95):
    """Середнє, с.к.в., мін, макс, 95% ДІ Стьюдента, відносна півширина."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "std": nan, "min": nan, "max": nan, "ci_lo": nan, "ci_hi": nan,
                "half": nan, "rel_half": nan}
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else float("nan")
    h = st.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "std": s, "min": x.min(), "max": x.max(),
            "ci_lo": m - h, "ci_hi": m + h, "half": h,
            "rel_half": h / abs(m) if m != 0 else float("nan")}


def chi2_normal_test(x, bins=None):
    """Критерій хі-квадрат Пірсона для нормальності: k рівноймовірних інтервалів,
    параметри оцінюються за вибіркою (df = k-1-2)."""
    x = np.asarray(x, float)
    n = len(x)
    k = bins or max(5, min(int(round(1.87 * (n - 1) ** 0.4)), n // 5))
    mu, sd = x.mean(), x.std(ddof=1)
    edges = st.norm.ppf(np.linspace(0, 1, k + 1), mu, sd)
    obs = np.histogram(x, bins=edges)[0]
    exp = n / k
    chi = ((obs - exp) ** 2 / exp).sum()
    df = k - 3
    return {"chi2": chi, "df": df, "p": st.chi2.sf(chi, df) if df > 0 else float("nan"), "k": k}


def identify(x, alpha=0.05):
    """Ідентифікація закону: підгонка кандидатів методом макс. правдоподібності, вибір за AIC, критерій
    Колмогорова-Смирнова; окремо Шапіро-Вілк і хі-квадрат для нормального закону."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    cands = {"normal": st.norm, "lognormal": st.lognorm, "gamma": st.gamma,
             "uniform": st.uniform, "expon": st.expon}
    rows = []
    for name, d in cands.items():
        try:
            if name in ("lognormal", "gamma", "expon") and x.min() <= 0:
                continue
            p = d.fit(x) if name not in ("lognormal", "gamma") else d.fit(x, floc=0)
            ll = d.logpdf(x, *p).sum()
            k = len(p) - (1 if name in ("lognormal", "gamma") else 0)
            aic = 2 * k - 2 * ll
            ks = st.kstest(x, d.cdf, args=p)
            rows.append({"law": name, "params": [float(v) for v in p], "aic": float(aic),
                         "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)})
        except Exception:
            pass
    rows.sort(key=lambda r: r["aic"])
    sw = st.shapiro(x) if len(x) >= 3 else None
    c2 = chi2_normal_test(x) if len(x) >= 20 else None
    out = {"best_aic": rows[0]["law"] if rows else None, "fits": rows,
           "shapiro_W": float(sw.statistic) if sw else None, "shapiro_p": float(sw.pvalue) if sw else None,
           "chi2": c2,
           "skew": float(st.skew(x)), "kurt_excess": float(st.kurtosis(x))}
    ok_n = (sw is not None and sw.pvalue > alpha) and (c2 is None or not (c2["p"] < alpha))
    out["normal_not_rejected"] = bool(ok_n)
    ukr = {"normal": "нормальний", "lognormal": "логнормальний", "gamma": "гамма", "uniform": "рівномірний",
           "expon": "показниковий"}
    best = ukr.get(rows[0]["law"], rows[0]["law"]) if rows else "невизначено"
    if ok_n:
        out["law"] = "нормальний"
    elif rows and rows[0]["law"] == "normal":
        out["law"] = "наближено нормальний (найкращий за AIC; тести відхиляють через дискретність даних)"
    else:
        out["law"] = best + " (нормальний відхилено тестом)"
    return out


def ols(x, y, conf=0.95):
    """Проста лінійна регресія y = a + b x: оцінки, СКП, ДІ, R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = st.linregress(x, y)
    t = st.t.ppf(0.5 + conf / 2, n - 2)
    return {"n": n, "slope": r.slope, "intercept": r.intercept, "r2": r.rvalue ** 2,
            "slope_se": r.stderr, "intercept_se": r.intercept_stderr,
            "slope_ci": (r.slope - t * r.stderr, r.slope + t * r.stderr),
            "intercept_ci": (r.intercept - t * r.intercept_stderr, r.intercept + t * r.intercept_stderr)}
