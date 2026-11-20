import json
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import d5

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SEED = 2026
METRICS = [("Pi", "Продуктивність Π, виробів/год", 0), ("Q23", "Черга до верстатів Q2+Q3", 1),
           ("Q1", "Черга заготовок Q1", 2), ("WIP", "Деталей у системі (WIP)", 3)]


def welch_ma(y, k):
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        h = min(i, k)
        lo, hi = i - h, i + h + 1
        if i + h + 1 > n:
            hi = n
            h2 = n - 1 - i
            lo = i - h2
        out[i] = y[lo:hi].mean()
    return out


def detect(ma, w, tol=0.05, tail=0.4, edge=0):
    """Перший момент, після якого рухоме середнє лишається в межах max(tol*рівень, 3*с.к.в. хвоста) навколо
    усталеного рівня; edge - кількість крайніх точок вікна, що не враховуються."""
    n = len(ma) - edge
    ma = ma[:n]
    tl = ma[int(n * (1 - tail)):]
    level, noise = tl.mean(), tl.std(ddof=1)
    band = max(tol * abs(level), 3 * noise)
    bad = np.where(np.abs(ma - level) > band)[0]
    j = (bad.max() + 1) if len(bad) else 0
    return j * w, level, band


NPROC = int(os.environ.get("NPROC", os.cpu_count() or 1))


def _series(a):
    tau, M, policy, horizon, w, key, r, proc_dist = a
    return d5.run_series(tau, M, policy, np.random.default_rng(np.random.SeedSequence([SEED, 1, key, r])),
                         horizon, w, proc_dist=proc_dist)


def run_case(tau, nruns, horizon, w, k, key, policy="weighted", M=3, proc_dist="exp", tag=""):
    jobs = [(tau, M, policy, horizon, w, key, r, proc_dist) for r in range(nruns)]
    if NPROC > 1:
        with Pool(NPROC) as p:
            ser = np.stack(p.map(_series, jobs, chunksize=2))
    else:
        ser = np.stack([_series(j) for j in jobs])
    mean = ser.mean(axis=0)              
    t = (np.arange(mean.shape[0]) + 0.5) * w
    df = pd.DataFrame({"t": t})
    res = {}
    for name, _, c in METRICS:
        ma = welch_ma(mean[:, c], k)
        df[name] = mean[:, c]
        df[name + "_MA"] = ma
        tp, level, band = detect(ma, w, edge=k)
        res[name] = {"T_trans": float(tp), "level": float(level), "band": float(band)}
    df.to_csv(os.path.join(RES, f"e1_welch_tau{tau}{tag}.csv"), index=False, float_format="%.6g")
    return df, res


CASES = [(20, 20000.0, 200.0, 5, 0), (14, 300000.0, 3000.0, 5, 1)]


def run_e1(nruns=200, proc_dist="exp", cases=None):
    out = {}
    for tau, hor, w, k, key in (cases or CASES):
        df, res = run_case(tau, nruns, hor, w, k, key, proc_dist=proc_dist)
        tp = max(v["T_trans"] for v in res.values())
        out[str(tau)] = {"proc_dist": proc_dist, "nruns": nruns, "horizon": hor, "window": w, "ma_halfwidth": k,
                         "per_metric": res, "T_trans_tau": tp}
        print(tau, {m: round(v["T_trans"]) for m, v in res.items()}, tp)
    out["T_trans_max"] = max(out["20"]["T_trans_tau"], out["14"]["T_trans_tau"])
    return out


if __name__ == "__main__":
    o = run_e1()
    json.dump(o, open(os.path.join(RES, "e1_summary.json"), "w"), ensure_ascii=False, indent=1)
