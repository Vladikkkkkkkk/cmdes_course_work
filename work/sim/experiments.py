import json
import math
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import d5
import transient
from stats import describe, identify, ols

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEED = 2026
NPROC = int(os.environ.get("NPROC", os.cpu_count() or 1))
KEYS_STAT = ["Pi_h", "U", "B", "Q1", "Q2", "Q3", "Q5", "Q6", "Q7", "Q8", "Q23",
             "max1", "max2", "max3", "max6", "lam1", "lam2", "lam3", "lam6",
             "W1", "W2", "W3", "W6", "Wd1", "Wd2", "Wd3", "Wd6", "Wmax1", "Wmax2", "Wmax3", "Wmax6",
             "busy_T2", "busy_T3", "busy_T4", "busy_T5", "N_prod", "sets"]


def jpath(n):
    return os.path.join(RES, n)


def load_params():
    p = {"T_warm": 3000.0, "T_stat": 50000.0, "N": 30, "proc_dist": "exp"}
    if os.path.exists(jpath("chosen_params.json")):
        p.update(json.load(open(jpath("chosen_params.json"))))
    return p


def _job(a):
    tau, M, pol, ids, tw, ts = a[:6]
    pdist = a[6] if len(a) > 6 else "exp"      
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *ids]))
    r = d5.run_model(tau, M, pol, rng, tw, ts, track=(1, 2, 3, 5, 6), proc_dist=pdist)
    r.pop("_sim", None)
    r.update({"tau": tau, "M": M, "policy": pol, "proc": pdist, "run": ids[-1]})
    return r


def pmap(jobs):
    if NPROC > 1:
        with Pool(NPROC) as p:
            return p.map(_job, jobs, chunksize=4)
    return [_job(j) for j in jobs]


def summarize(df, keys=KEYS_STAT):
    rows = []
    for k in keys:
        if k not in df:
            continue
        d = describe(df[k])
        rows.append({"metric": k, **d})
    return pd.DataFrame(rows)


def e1():
    o = transient.run_e1()
    tmax = o["T_trans_max"]
    o["T_warm_chosen"] = float(math.ceil(1.5 * tmax / 500.0) * 500.0)
    o["rule"] = "T_пер = 1.5 * max(T_пер за Велчем для tau=20 та 14), округлено вгору до 500 хв; обробка Exp (база), політика weighted (база)"
    json.dump(o, open(jpath("e1_summary.json"), "w"), ensure_ascii=False, indent=1)
    p = load_params()
    p["T_warm"] = o["T_warm_chosen"]
    json.dump(p, open(jpath("chosen_params.json"), "w"), ensure_ascii=False, indent=1)
    print("T_warm =", p["T_warm"])



N_E3E4 = 45
T_STAT_ACCEPTED = 200000.0   


def stable_nmin(hh, col, lim):
    """СТІЙКЕ N_min: найменше n, починаючи з якого відн. півширина <= lim для ВСІХ більших n (до nmax)."""
    hh = hh.sort_values("n")
    ok = (hh[col] <= lim).values
    ns = hh["n"].values
    for i in range(len(ns)):
        if ok[i:].all():
            return int(ns[i])
    return None


def _e2_table(hw, grid):
    """Правило: Pi, U, Q1 <= 3% (обидва tau); Q2+Q3 <= 5% (tau=20) і <= 10% (tau=14, rho=0,95);
    N_min -- стійке (див. stable_nmin). T_ст фіксовано (T_STAT_ACCEPTED), N = max стійких N_min."""
    crit = {"Pi_h": 0.03, "U": 0.03, "Q1": 0.03}
    table = []
    for ts in grid:
        h = hw[hw["T_stat"] == ts]
        nmin = {}
        for tau in (20, 14):
            hh = h[h.tau == tau]
            for k, lim in crit.items():
                nmin[f"{tau}_{k}"] = stable_nmin(hh, f"rel_half_{k}", lim)
        for tau, lim in ((20, 0.05), (14, 0.10)):
            nmin[f"{tau}_Q23"] = stable_nmin(h[h.tau == tau], "rel_half_Q23", lim)
        worst = max([v for v in nmin.values() if v is not None] + [0])
        feasible = all(v is not None for v in nmin.values())
        table.append({"T_stat": ts, "N_min": nmin, "N_min_worst": worst, "feasible": feasible})
    return T_STAT_ACCEPTED, table


def table_N(table, ts):
    return int([t for t in table if t["T_stat"] == ts][0]["N_min_worst"])


def e2tab():
    p = load_params()
    hw = pd.read_csv(jpath("e2_halfwidth.csv"))
    grid = sorted(hw["T_stat"].unique())
    chosen, table = _e2_table(hw, grid)
    p.update({"T_stat": chosen, "N": max(N_E3E4, table_N(table, chosen)), "N_stable_E2": table_N(table, chosen), "N_E0": 100, "N_verify": 30,
              "T_stat_verify": chosen, "E2_table": table})
    json.dump(p, open(jpath("chosen_params.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("T_stat =", chosen, "N =", p["N"])
    for t in table:
        print(t)

def e2():
    p = load_params()
    tw = p["T_warm"]
    grid = [5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 200000.0, 400000.0]
    nmax = 100
    jobs = [(tau, 3, "weighted", (2, ti, gi, r), tw, ts)
            for ti, tau in enumerate((20, 14)) for gi, ts in enumerate(grid) for r in range(nmax)]
    res = pmap(jobs)
    df = pd.DataFrame(res)
    df["T_stat"] = [g for tau in (20, 14) for g in grid for _ in range(nmax)]
    df.to_csv(jpath("e2_runs.csv"), index=False, float_format="%.6g")
    rows = []
    for (tau, ts), g in df.groupby(["tau", "T_stat"]):
        g = g.sort_values("run")
        for n in range(5, nmax + 1):
            row = {"tau": tau, "T_stat": ts, "n": n}
            for k in ("Pi_h", "U", "Q1", "Q23"):
                row[f"rel_half_{k}"] = describe(g[k].iloc[:n])["rel_half"]
            rows.append(row)
    hw = pd.DataFrame(rows)
    hw.to_csv(jpath("e2_halfwidth.csv"), index=False, float_format="%.6g")
    chosen, table = _e2_table(hw, grid)
    p.update({"T_stat": chosen, "N": max(N_E3E4, table_N(table, chosen)), "N_stable_E2": table_N(table, chosen), "N_E0": 100, "N_verify": 30,
              "T_stat_verify": chosen, "E2_table": table})
    json.dump(p, open(jpath("chosen_params.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("T_stat =", chosen)
    for t in table:
        print(t)


def e0():
    p = load_params()
    tw, ts, n = p["T_warm"], p["T_stat"], int(p.get("N_E0", 100))
    res = pmap([(20, 3, "weighted", (0, 0, r), tw, ts) for r in range(n)])
    df = pd.DataFrame(res)
    df.to_csv(jpath("e0_runs.csv"), index=False, float_format="%.6g")
    S = summarize(df)
    ids = {}
    for k in S.metric:
        x = df[k].to_numpy()
        if np.nanstd(x) < 1e-12:
            ids[k] = {"law": "константа (виродкий розподіл)", "best_aic": None}
            continue
        i = identify(x)
        ids[k] = {"law": i["law"], "best_aic": i["best_aic"], "shapiro_p": i["shapiro_p"],
                  "chi2_p": i["chi2"]["p"] if i["chi2"] else None, "skew": i["skew"],
                  "kurt_excess": i["kurt_excess"]}
    S["law"] = [ids[k]["law"] for k in S.metric]
    S["best_aic"] = [ids[k]["best_aic"] for k in S.metric]
    S["shapiro_p"] = [ids[k].get("shapiro_p") for k in S.metric]
    S["chi2_p"] = [ids[k].get("chi2_p") for k in S.metric]
    S.to_csv(jpath("e0_summary.csv"), index=False, float_format="%.6g")
    json.dump({"params": {"tau": 20, "M": 3, "policy": "weighted", "T_warm": tw, "T_stat": ts, "N": n},
               "identification": ids}, open(jpath("e0_identification.json"), "w"), ensure_ascii=False,
              indent=1, default=float)
    samp = {1: [], 2: [], 3: [], 6: []}
    for r in range(5):
        rr = d5.run_model(20, 3, "weighted", np.random.default_rng(np.random.SeedSequence([SEED, 0, 1, r])),
                          tw, ts, track=(1, 2, 3, 6), keep_samples=True)
        for k in samp:
            samp[k] += rr["_sim"].samples[k]
    wrows = []
    for k, v in samp.items():
        v = np.array(v)
        nz = v[v > 1e-9]
        d = describe(v)
        wrows.append({"place": f"P{k}", "n": len(v), "mean": d["mean"], "std": d["std"], "min": d["min"],
                      "max": d["max"], "frac_zero": float((v <= 1e-9).mean()),
                      "mean_nonzero": float(nz.mean()) if len(nz) else float("nan"),
                      "best_aic_nonzero": identify(nz)["best_aic"] if len(nz) > 30 else None})
    pd.DataFrame(wrows).to_csv(jpath("e0_wait_tokens.csv"), index=False, float_format="%.6g")
    print(S[["metric", "mean", "std", "min", "max", "ci_lo", "ci_hi", "law"]].to_string())


TAUS3 = [8, 10, 11, 12, 12.5, 13, 13.33, 13.5, 14, 15, 16, 18, 20, 25, 30]


def theory_pi(tau, M=3, policy="weighted"):
    """Теоретична продуктивність (виробів/год): 20/tau у стійкій зоні; у перевантаженні плато M/2 (збалансована
    політика) або M - lam (строгий пріоритет типу 1)."""
    lam = 20.0 / tau                                   
    if lam < M / 2.0:
        return lam
    if policy == "weighted":
        return M / 2.0
    return max(0.0, M - lam)


def e3():
    p = load_params()
    tw, ts, n = p["T_warm"], p["T_stat"], int(p["N"])
    jobs = [(tau, 3, "weighted", (3, ti, r), tw, ts) for ti, tau in enumerate(TAUS3) for r in range(n)]
    df = pd.DataFrame(pmap(jobs))
    df.to_csv(jpath("e3_runs.csv"), index=False, float_format="%.6g")
    rows = []
    for tau, g in df.groupby("tau"):
        for k in KEYS_STAT:
            d = describe(g[k])
            rows.append({"tau": tau, "metric": k, **d})
    S = pd.DataFrame(rows)
    S["stationary"] = S.tau > 40.0 / 3
    S.to_csv(jpath("e3_summary.csv"), index=False, float_format="%.6g")
    piS = S[S.metric == "Pi_h"].set_index("tau")
    st_ = df[df.tau >= 14].copy()
    x = 1.0 / st_.tau.to_numpy()
    reg = ols(x, st_.Pi_h.to_numpy())
    mean_pts = piS.loc[[t for t in TAUS3 if t >= 14], "mean"]
    reg_means = ols(1.0 / mean_pts.index.to_numpy(float), mean_pts.to_numpy())
    tau_peak = float(piS["mean"].idxmax())
    Uc = S[S.metric == "U"].set_index("tau")["mean"]
    tau_u99 = float(max(t for t in TAUS3 if Uc[t] >= 0.99))
    out = {"regression_all_runs_stable_zone": reg, "regression_on_means": reg_means,
           "theory_slope": 20.0, "tau_peak_Pi_sim": tau_peak, "Pi_peak_sim": float(piS["mean"].max()),
           "tau_star_theory": 40.0 / 3, "Pi_star_theory_h": 1.5,
           "tau_largest_with_U_ge_0.99": tau_u99,
           "tau_star_from_fit": reg["slope"] / float(piS["mean"].max()),
           "params": {"T_warm": tw, "T_stat": ts, "N": n}}
    json.dump(out, open(jpath("e3_regression.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


def _stage(name, jobs, fn):
    parts = os.environ.get("E4_PARTS")
    if parts is not None and name not in parts.split(","):
        return pd.read_csv(jpath(fn))
    sl = os.environ.get("E4_SLICE")   
    if sl:
        i, k = sl.split("/")
        k = int(k)
        if i == "merge":
            df = pd.concat([pd.read_csv(jpath(f"{fn}.part{j}")) for j in range(k)], ignore_index=True)
        else:
            pd.DataFrame(pmap(jobs[int(i)::k])).to_csv(jpath(f"{fn}.part{i}"), index=False, float_format="%.6g")
            raise SystemExit(0)
    else:
        df = pd.DataFrame(pmap(jobs))
    df.to_csv(jpath(fn), index=False, float_format="%.6g")
    return df


def e4():
    p = load_params()
    tw, ts, n = p["T_warm"], p["T_stat"], int(p["N"])
    taus_m = [8, 10, 12, 14, 16, 20]
    Ms = [2, 3, 4, 5]
    jobs = [(tau, M, "weighted", (4, 0, mi, ti, r), tw, ts)
            for mi, M in enumerate(Ms) for ti, tau in enumerate(taus_m) for r in range(n)]
    dM = _stage("M", jobs, "e4_M_runs.csv")
    rows = []
    for (M, tau), g in dM.groupby(["M", "tau"]):
        for k in ["Pi_h", "U", "Q1", "Q2", "Q3", "Q6", "Wd2", "Wd3", "max3"]:
            rows.append({"M": M, "tau": tau, "metric": k, **describe(g[k]), "pi_theory": theory_pi(tau, M) if k == "Pi_h" else np.nan})
    SM = pd.DataFrame(rows)
    SM.to_csv(jpath("e4_M_summary.csv"), index=False, float_format="%.6g")
    pols = ["weighted", "priority", "priority3", "random"]
    taus_p = [8, 10, 12, 13, 14, 16, 20]
    jobs = [(tau, 3, pol, (4, 1, pi, ti, r), tw, ts)
            for pi, pol in enumerate(pols) for ti, tau in enumerate(taus_p) for r in range(n)]
    dP = _stage("P", jobs, "e4_policy_runs.csv")
    rows = []
    for (pol, tau), g in dP.groupby(["policy", "tau"]):
        for k in ["Pi_h", "U", "Q1", "Q2", "Q3", "Q6", "Q7", "Q8", "Wd2", "Wd3", "max2", "max3"]:
            rows.append({"policy": pol, "tau": tau, "metric": k, **describe(g[k])})
    SP = pd.DataFrame(rows)
    SP.to_csv(jpath("e4_policy_summary.csv"), index=False, float_format="%.6g")
    
    taus_d = [20, 14, 12]
    pdists = ["exp", "det"]
    jobs = [(tau, 3, "weighted", (4, 2, di, ti, r), tw, ts, pdist)
            for di, pdist in enumerate(pdists) for ti, tau in enumerate(taus_d) for r in range(n)]
    dD = _stage("D", jobs, "e4_proc_runs.csv")
    kD = ["Pi_h", "U", "Q1", "Q2", "Q3", "Q6", "Wd2", "Wd3", "max3"]
    rows = []
    for tau in taus_d:
        for k in kD:
            de = describe(dD[(dD.tau == tau) & (dD.proc == "exp")][k])
            dd = describe(dD[(dD.tau == tau) & (dD.proc == "det")][k])
            xe = dD[(dD.tau == tau) & (dD.proc == "exp")][k].to_numpy()
            xd = dD[(dD.tau == tau) & (dD.proc == "det")][k].to_numpy()
            from scipy import stats as sst
            pv = float(sst.ttest_ind(xe, xd, equal_var=False).pvalue) if (xe.std() > 0 or xd.std() > 0) else float("nan")
            rows.append({"tau": tau, "metric": k, "exp_mean": de["mean"], "exp_ci_lo": de["ci_lo"], "exp_ci_hi": de["ci_hi"],
                         "det_mean": dd["mean"], "det_ci_lo": dd["ci_lo"], "det_ci_hi": dd["ci_hi"],
                         "ratio_exp_det": de["mean"] / dd["mean"] if dd["mean"] else float("nan"),
                         "welch_p": pv})
    SD = pd.DataFrame(rows)
    SD.to_csv(jpath("e4_proc_summary.csv"), index=False, float_format="%.6g")


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    steps = {"e1": e1, "e2": e2, "e2tab": e2tab, "e0": e0, "e3": e3, "e4": e4}
    for w in (list(steps) if what == ["all"] else what):
        print("==", w, flush=True)
        steps[w]()
