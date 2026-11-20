"""Додаткові прогони (діагностика): python3 wave2.py s1 [T ...] | s2 | s3 | an. s1 - недобір плато при tau=8 залежно від
T_ст; s2 - вплив T_пер при tau=13,33; s3 - M=3 і M=4 за строгого пріоритету типу 1; an - зведення (../results/w2_*.csv)."""
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import d5
from stats import describe

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEED = 2026
TW = 27000.0
NPROC = int(os.environ.get("NPROC", os.cpu_count() or 1))


def _job(a):
    tau, M, pol, tw, ts, ids = a
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, *ids]))
    r = d5.run_model(tau, M, pol, rng, tw, ts, track=())
    r.pop("_sim", None)
    r.update({"tau": tau, "M": M, "policy": pol, "T_warm": tw, "T_stat": ts, "run": ids[-1]})
    return r


def pm(jobs):
    with Pool(NPROC) as p:
        return pd.DataFrame(p.map(_job, jobs, chunksize=2))


def s1(Ts):
    for ti, ts in enumerate(Ts):
        n = 100
        df = pm([(8, 3, "weighted", TW, ts, (0, int(ts), r)) for r in range(n)])
        df.to_csv(os.path.join(RES, f"w2_s1_T{int(ts)}.csv"), index=False, float_format="%.8g")
        print("s1", ts, df.Pi_h.mean(), flush=True)


def s2():
    n = 60
    for tw in (500000.0, TW):
        df = pm([(13.33, 3, "weighted", tw, 200000.0, (1, int(tw), r)) for r in range(n)])
        df.to_csv(os.path.join(RES, f"w2_s2_tw{int(tw)}.csv"), index=False, float_format="%.8g")
        print("s2", tw, df.Pi_h.mean(), df.U.mean(), flush=True)


def s3():
    n = 30
    jobs = [(tau, M, "priority", TW, 200000.0, (3, mi, ti, r))
            for mi, M in enumerate((3, 4)) for ti, tau in enumerate((8, 10, 12)) for r in range(n)]
    df = pm(jobs)
    df.to_csv(os.path.join(RES, "w2_s3_M_priority.csv"), index=False, float_format="%.8g")
    print("s3 done")


def imb(df):
    T = df["T"].iloc[0]
    n1 = df.lam5 * T
    n2 = df.lam6 * T
    return n1, n2 / 2.0, T


def an():
    out = {}
    rows = []
    for f in sorted(os.listdir(RES)):
        if f.startswith("w2_s1_T"):
            df = pd.read_csv(os.path.join(RES, f))
            n1, pr, T = imb(df)
            S = T / 40.0                                  
            d = describe(df.Pi_h)
            absd = (n1 - pr).abs()
            pi_min = (np.minimum(n1, pr) * 60.0 / T)
            rows.append({"T_stat": T, "N": len(df), "Pi_mean": d["mean"], "Pi_half": d["half"],
                         "deficit_rel": (1.5 - d["mean"]) / 1.5,
                         "deficit_rel_half": d["half"] / 1.5,
                         "E_abs_n1_minus_n2half": absd.mean(),
                         "pred_rel": absd.mean() / (2.0 * S),
                         "Pi_from_min": pi_min.mean(),
                         "deficit_rel_from_min": (1.5 - pi_min.mean()) / 1.5,
                         "mean_n1_minus_n2half": (n1 - pr).mean(), "U_mean": df.U.mean()})
    s1 = pd.DataFrame(rows).sort_values("T_stat")
    s1["ratio_meas_pred"] = s1.deficit_rel / s1.pred_rel
    cbar = float((s1.E_abs_n1_minus_n2half / np.sqrt(s1.T_stat)).mean())
    s1["pred2_rel"] = cbar * (np.sqrt(TW + s1.T_stat) - np.sqrt(TW)) / (2.0 * s1.T_stat / 40.0)
    s1["ratio_meas_pred2"] = s1.deficit_rel / s1.pred2_rel
    out["s1_c_hat"] = cbar
    out["s1_chi2_pred"] = float((((s1.deficit_rel - s1.pred_rel) / (s1.deficit_rel_half / 1.96)) ** 2).sum())
    out["s1_chi2_pred2"] = float((((s1.deficit_rel - s1.pred2_rel) / (s1.deficit_rel_half / 1.96)) ** 2).sum())
    out["s1_df"] = int(len(s1))
    s1.to_csv(os.path.join(RES, "w2_s1_summary.csv"), index=False, float_format="%.6g")
    if len(s1) >= 3:
        x, y = np.log(s1.T_stat.to_numpy()), np.log(s1.deficit_rel.to_numpy())
        b = np.polyfit(x, y, 1)
        yp = np.log(s1.pred_rel.to_numpy())
        bp = np.polyfit(x, yp, 1)
        out["s1_loglog_slope_deficit"] = float(b[0])
        out["s1_loglog_slope_pred"] = float(bp[0])
    s2 = []
    for tw in (500000, 27000):
        p = os.path.join(RES, f"w2_s2_tw{tw}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            n1, pr, T = imb(df)
            for k in ("Pi_h", "U", "Q2", "Q3", "Q7", "Q8"):
                d = describe(df[k])
                s2.append({"T_warm": tw, "metric": k, "mean": d["mean"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": len(df)})
            s2.append({"T_warm": tw, "metric": "E|n1-n2/2|", "mean": (n1 - pr).abs().mean(), "ci_lo": np.nan, "ci_hi": np.nan, "n": len(df)})
            s2.append({"T_warm": tw, "metric": "pred_rel_deficit", "mean": (n1 - pr).abs().mean() / (2.0 * T / 40.0), "ci_lo": np.nan, "ci_hi": np.nan, "n": len(df)})
    pd.DataFrame(s2).to_csv(os.path.join(RES, "w2_s2_summary.csv"), index=False, float_format="%.6g")
    p = os.path.join(RES, "w2_s3_M_priority.csv")
    if os.path.exists(p):
        pri = pd.read_csv(p)
        wt = pd.read_csv(os.path.join(RES, "e4_M_runs.csv"))
        rows = []
        for pol, dd in (("weighted", wt), ("priority", pri)):
            for tau in (8, 10, 12):
                a = dd[(dd.M == 3) & (dd.tau == tau)].Pi_h.to_numpy()
                b = dd[(dd.M == 4) & (dd.tau == tau)].Pi_h.to_numpy()
                m3, m4 = a.mean(), b.mean()
                se3, se4 = a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))
                r = m4 / m3
                se_r = r * np.sqrt((se3 / m3) ** 2 + (se4 / m4) ** 2)
                rows.append({"policy": pol, "tau": tau, "Pi_M3": m3, "Pi_M4": m4, "gain_pct": (r - 1) * 100,
                             "gain_pct_ci95_half": 2.0 * se_r * 100, "n3": len(a), "n4": len(b)})
        pd.DataFrame(rows).to_csv(os.path.join(RES, "w2_s3_summary.csv"), index=False, float_format="%.6g")
    json.dump(out, open(os.path.join(RES, "w2_summary.json"), "w"), indent=1)
    print(s1.round(5).to_string())
    print(pd.DataFrame(s2).round(5).to_string() if s2 else "no s2")
    print(out)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "s1":
        s1([float(t) for t in sys.argv[2:]] or [25000.0, 50000.0, 100000.0, 200000.0, 400000.0, 800000.0])
    elif c == "s2":
        s2()
    elif c == "s3":
        s3()
    else:
        an()
