import json
import os
import sys
from collections import deque
import heapq
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def _overlap(a, b, t0, t1):
    a, b = np.asarray(a), np.asarray(b)
    return np.clip(np.minimum(b, t1) - np.maximum(a, t0), 0, None).sum()


def simulate(tau, M, policy, rng, t_warm, t_stat, width=5.0, proc_dist="exp"):
    t0, t1 = t_warm, t_warm + t_stat
    n_arr = int(t1 / (tau - width) / 1.0) + 20
    n_arr += (-n_arr) % 3
    cum = np.cumsum(rng.uniform(tau - width, tau + width, n_arr))
    set_t = cum[2::3]
    set_t = set_t[set_t <= t1]
    n_sets = len(set_t)
    # ---- P1: заготовка i чекає до кінця свого комплекту
    n_tok = 3 * n_sets
    tok_a = cum[:n_tok]
    tok_b = np.repeat(set_t, 3)
    # ---- верстати
    free = [0.0] * M
    heapq.heapify(free)
    q = {1: deque(), 2: deque()}
    log = {1: ([], [], []), 2: ([], [], [])}    
    idx, clock = 0, 0.0
    INF = float("inf")
    p3 = {"priority": None, "priority3": None, "random": 0.5, "weighted": 2 / 3}[policy]

    def add_set(s):
        q[1].append(s)
        q[2].append(s)
        q[2].append(s)

    while True:
        nxt = set_t[idx] if idx < n_sets else INF
        if q[1] or q[2]:
            f = free[0]
            start = max(f, clock)
            if nxt <= start:
                clock = max(clock, nxt)
                add_set(nxt)
                idx += 1
                continue
            if start > t1:
                break
            if q[1] and q[2]:
                if policy == "priority":
                    typ = 1
                elif policy == "priority3":
                    typ = 2
                else:
                    typ = 2 if rng.random() < p3 else 1
            else:
                typ = 1 if q[1] else 2
            arr = q[typ].popleft()
            heapq.heappop(free)
            mean_s = 60.0 if typ == 1 else 30.0
            fin = start + (rng.exponential(mean_s) if proc_dist == "exp" else mean_s)
            heapq.heappush(free, fin)
            clock = start
            L = log[typ]
            L[0].append(arr); L[1].append(start); L[2].append(fin)
        else:
            if nxt == INF or nxt > t1:
                break
            clock = max(clock, nxt)
            add_set(nxt)
            idx += 1
    rest1, rest2 = np.array(q[1], float), np.array(q[2], float)
    a1, s1, f1 = (np.array(v) for v in log[1])
    a2, s2, f2 = (np.array(v) for v in log[2])
    busy = _overlap(s1, f1, t0, t1) + _overlap(s2, f2, t0, t1)
    # ---- зборка і вироби
    x = np.sort(f1 + rng.uniform(58, 62, len(f1)))
    v = np.sort(f2)
    npair = len(v) // 2
    pair_first, pair_second = v[0:2 * npair:2], v[1:2 * npair:2]
    y = np.sort(pair_second + rng.uniform(52, 68, npair))
    k = min(len(x), len(y))
    prod_t = np.maximum(x[:k], y[:k])
    n_prod = int(((prod_t > t0) & (prod_t <= t1)).sum())
    return {"Pi_h": n_prod / t_stat * 60.0,
            "U": busy / (M * t_stat),
            "Q1": (_overlap(tok_a, tok_b, t0, t1)
                   + _overlap(cum[n_tok:][cum[n_tok:] <= t1], np.full(int((cum[n_tok:] <= t1).sum()), t1), t0, t1)) / t_stat,
            "Q2": (_overlap(a1, s1, t0, t1) + _overlap(rest1, np.full(len(rest1), t1), t0, t1)) / t_stat,
            "Q3": (_overlap(a2, s2, t0, t1) + _overlap(rest2, np.full(len(rest2), t1), t0, t1)) / t_stat,
            "Q6": _overlap(pair_first, pair_second, t0, t1) / t_stat}


CONFIGS = [  
    (20, 3, "weighted"), (14, 3, "weighted"), (20, 4, "weighted"),
    (20, 3, "priority"), (14, 3, "random"), (14, 3, "priority"),
    (12, 3, "weighted"),   
]
KEYS = ["Pi_h", "U", "Q1", "Q2", "Q3", "Q6"]


def _cfg_job(a):
    import d5
    ci, (tau, M, pol), n_runs, t_warm, t_stat, seed, proc_dist = a
    A, B = [], []
    for r in range(n_runs):
        ra = np.random.default_rng(np.random.SeedSequence([seed, 6, ci, r, 0]))
        rb = np.random.default_rng(np.random.SeedSequence([seed, 6, ci, r, 1]))
        A.append(d5.run_model(tau, M, pol, ra, t_warm, t_stat, track=(), proc_dist=proc_dist))
        B.append(simulate(tau, M, pol, rb, t_warm, t_stat, proc_dist=proc_dist))
    rows = []
    for k in KEYS:
        xa = np.array([a_[k] for a_ in A]); xb = np.array([b_[k] for b_ in B])
        tt = st.ttest_ind(xa, xb, equal_var=False)
        sa, sb = xa.std(ddof=1), xb.std(ddof=1)
        hw = lambda s_: st.t.ppf(0.975, n_runs - 1) * s_ / np.sqrt(n_runs)
        rows.append({"tau": tau, "M": M, "policy": pol, "metric": k,
                     "spn_mean": xa.mean(), "spn_half": hw(sa),
                     "ind_mean": xb.mean(), "ind_half": hw(sb),
                     "diff": xa.mean() - xb.mean(),
                     "welch_t": float(tt.statistic), "p": float(tt.pvalue),
                     "agree_p05": bool(tt.pvalue > 0.05),
                     "ci_overlap": bool(abs(xa.mean() - xb.mean()) <= hw(sa) + hw(sb))})
    return rows


def compare(n_runs=30, t_warm=2000.0, t_stat=20000.0, seed=2026, proc_dist="exp"):
    from multiprocessing import Pool
    nproc = int(os.environ.get("NPROC", os.cpu_count() or 1))
    jobs = [(ci, c, n_runs, t_warm, t_stat, seed, proc_dist) for ci, c in enumerate(CONFIGS)]
    if nproc > 1:
        with Pool(nproc) as p:
            out = p.map(_cfg_job, jobs, chunksize=1)
    else:
        out = [_cfg_job(j) for j in jobs]
    return [r for rows in out for r in rows]


if __name__ == "__main__":
    import pandas as pd
    p = {}
    pj = os.path.join(RES, "chosen_params.json")
    if os.path.exists(pj):
        p = json.load(open(pj))
    n = int(p.get("N_verify", 30))
    tw, ts = float(p.get("T_warm", 2000)), float(p.get("T_stat_verify", 20000))
    pdist = sys.argv[1] if len(sys.argv) > 1 else "exp"
    rows = compare(n, tw, ts, proc_dist=pdist)
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    sfx = "" if pdist == "exp" else "_" + pdist     
    df.to_csv(os.path.join(RES, f"v6_independent{sfx}.csv"), index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(df[["tau", "M", "policy", "metric", "spn_mean", "ind_mean", "diff", "p", "ci_overlap"]].to_string())
    summ = {"proc_dist": pdist, "n_runs": n, "t_warm": tw, "t_stat": ts, "n_comparisons": len(df),
            "n_p_below_005": int((df.p < 0.05).sum()),
            "expected_by_chance": round(0.05 * len(df), 1),
            "min_p": float(df.p.min()), "n_ci_overlap": int(df.ci_overlap.sum())}
    json.dump(summ, open(os.path.join(RES, f"v6_summary{sfx}.json"), "w"), ensure_ascii=False, indent=1)
    print(summ)
