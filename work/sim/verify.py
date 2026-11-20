"""Верифікація моделі D5 та ядра СМП: V1-V5 (V6 - independent_check.py). Результати - у work/results/."""
import json
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import d5
from spn import Simulator
from stats import describe

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
P = {"T_warm": 2000.0, "T_stat": 20000.0, "N_verify": 30}
pj = os.path.join(RES, "chosen_params.json")
if os.path.exists(pj):
    P.update(json.load(open(pj)))
TW, TS, NV = float(P["T_warm"]), float(P.get("T_stat_verify", P["T_stat"])), int(P.get("N_verify", 30))
SEED = 2026
NPROC = int(os.environ.get("NPROC", os.cpu_count() or 1))


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))



def _row(t, mu, act):
    return (float(t), tuple(mu.get(i, 0) for i in range(10)), tuple(act.get(i, 0) for i in range(7)))


_A9 = {0: 1, 2: 1, 3: 2, 4: 1, 5: 1}  
_A11 = {0: 1, 2: 1, 4: 1, 5: 1}
_A13 = {0: 1, 1: 1, 4: 1, 5: 1, 6: 1}
EXPECTED = [
    _row(0, {4: 3}, {0: 1}),
    _row(20, {1: 1, 4: 3}, {0: 1}),
    _row(40, {1: 2, 4: 3}, {0: 1}),
    _row(60, {4: 3}, {0: 1, 1: 1}),
    _row(60, {}, {0: 1, 2: 1, 3: 2}),
    _row(80, {1: 1}, {0: 1, 2: 1, 3: 2}),
    _row(90, {1: 1, 4: 2}, {0: 1, 2: 1, 5: 1}),
    _row(100, {1: 2, 4: 2}, {0: 1, 2: 1, 5: 1}),
    _row(120, {4: 3}, {0: 1, 1: 1, 4: 1, 5: 1}),
    _row(120, {}, _A9),
    _row(140, {1: 1}, _A9),
    _row(150, {1: 1, 4: 2, 8: 1}, _A11),
    _row(160, {1: 2, 4: 2, 8: 1}, _A11),
    _row(180, {4: 3}, _A13),
    _row(180, {9: 1}, _A9),
    _row(200, {1: 1, 9: 1}, _A9),
    _row(210, {1: 1, 4: 2, 8: 1, 9: 1}, _A11),
    _row(220, {1: 2, 4: 2, 8: 1, 9: 1}, _A11),
    _row(240, {4: 3, 9: 1}, _A13),
    _row(240, {9: 2}, _A9),
    _row(260, {1: 1, 9: 2}, _A9),
    _row(270, {1: 1, 4: 2, 8: 1, 9: 2}, _A11),
    _row(280, {1: 2, 4: 2, 8: 1, 9: 2}, _A11),
    _row(300, {4: 3, 9: 2}, _A13),
    _row(300, {9: 3}, _A9),
    _row(320, {1: 1, 9: 3}, _A9),
    _row(330, {1: 1, 4: 2, 8: 1, 9: 3}, _A11),
    _row(340, {1: 2, 4: 2, 8: 1, 9: 3}, _A11),
    _row(360, {4: 3, 9: 3}, _A13),
    _row(360, {9: 4}, _A9),
]


def v1():
    trace = []
    sim = Simulator(d5.build_net(20.0, 3, "priority", deterministic=True), np.random.default_rng(0),
                    trace=trace)
    while len(trace) < len(EXPECTED):
        sim.step()
    ok = all(tr == ex for tr, ex in zip(trace, EXPECTED))
    lines = ["# V1. Детермінована траса: ядро СМП проти ручного розрахунку\n",
             "Усі затримки сталі: T0=20, T2=60, T3=30, T4=T5=60, M=3, пріоритет T2>T3. "
             "Рядок - стан після ітерації алгоритму: час, маркування (P0..P9, лише ненульові), "
             "активні канали (T0..T6). `ok` - збіг ядра з ручним розрахунком у `verify.py`.\n",
             "| n | t, хв | маркування | активні канали | ok |", "|---|---|---|---|---|"]
    for n, (tr, ex) in enumerate(zip(trace, EXPECTED)):
        mu = ", ".join(f"P{i}={v}" for i, v in enumerate(tr[1]) if v) or "усі 0"
        ac = ", ".join(f"T{i}={v}" for i, v in enumerate(tr[2]) if v)
        lines.append(f"| {n} | {tr[0]:g} | {mu} | {ac} | {'так' if tr == ex else 'НІ'} |")
    lines.append(f"\nПерші {len(EXPECTED)} ітерацій збігаються: **{ok}**. Перший виріб - t=180, далі по одному "
                 "кожні 60 хв (Π=1 виріб/год, як у теорії для tau=20).")
    open(os.path.join(RES, "v1_trace.md"), "w").write("\n".join(lines) + "\n")
    sim2 = Simulator(d5.build_net(20.0, 3, "priority", deterministic=True), np.random.default_rng(0),
                     check=d5.make_checker(3))
    sim2.run(600.0)
    sim2.reset_stats(600.0)
    sim2.run(600.0 + 6000.0)
    r = d5.collect(sim2, 6000.0)
    return ok, {"Pi_h": r["Pi_h"], "U": r["U"], "Q1": r["Q1"], "Q3": r["Q3"]}


# ---------------------------------------------------------------- V2
def v2():
    rows = []
    total_iter = 0
    for ci, (tau, M, pol) in enumerate([(20, 3, "weighted"), (14, 3, "weighted"), (12, 3, "priority"),
                                        (8, 3, "random"), (10, 3, "priority"), (20, 4, "priority3"),
                                        (14, 2, "weighted"), (30, 5, "random"), (20, 3, "weighted:det")]):
        pol, _, pdist = pol.partition(":")
        sim = Simulator(d5.build_net(tau, M, pol, proc_dist=pdist or "exp"), rng_for(2, ci), check=d5.make_checker(M))
        sim.run(30000.0)          
        total_iter += sim.n_iter
        rows.append({"tau": tau, "M": M, "policy": pol, "proc_dist": pdist or "exp", "iterations": sim.n_iter, "violations": 0})
    sim = Simulator(d5.build_net(20, 3, "priority"), rng_for(2, 99), check=None)
    sim.run(500.0)
    detected = {}
    for name, fn in {"зайвий вільний верстат": lambda s: s.mu.__setitem__(4, s.mu[4] + 1),
                     "зникла заготовка": lambda s: s.mu.__setitem__(1, s.mu[1] + 1),
                     "зайвий виріб": lambda s: s.mu.__setitem__(9, s.mu[9] + 1)}.items():
        s2 = Simulator(d5.build_net(20, 3, "priority"), rng_for(2, 98))
        s2.run(500.0)
        fn(s2)
        try:
            d5.make_checker(3)(s2)
            detected[name] = False
        except AssertionError:
            detected[name] = True
    out = {"configs": rows, "total_iterations_checked": total_iter, "all_ok": True,
           "tamper_detected": detected}
    json.dump(out, open(os.path.join(RES, "v2_invariants.json"), "w"), ensure_ascii=False, indent=1)
    return all(detected.values()), out


def _bjob(a):
    tau, M, pol, key, r, tw, ts, track, pdist = a
    return {k: v for k, v in d5.run_model(tau, M, pol, rng_for(*key, r), tw, ts, track=track,
                                          proc_dist=pdist).items() if k != "_sim"}


def batch(tau, M, pol, key, n=NV, tw=TW, ts=TS, track=(1, 2, 3, 5, 6), pdist="exp"):
    jobs = [(tau, M, pol, key, r, tw, ts, track, pdist) for r in range(n)]
    if NPROC > 1:
        with Pool(NPROC) as p:
            return pd.DataFrame(p.map(_bjob, jobs, chunksize=2))
    return pd.DataFrame([_bjob(j) for j in jobs])


def theory_stable(tau, M=3):
    lam = 1.0 / (3 * tau)       
    return {"Pi_h": 60 * lam, "U": 120 * lam / M, "B": 120 * lam, "Q1": 1.0, "Q5": 0.0,
            "lam1": 1.0 / tau, "lam2": lam, "lam3": 2 * lam, "W1": tau,
            "busy_T2": 60 * lam, "busy_T3": 60 * lam, "busy_T4": 60 * lam, "busy_T5": 60 * lam}


def v3():
    rows = []
    for ti, tau in enumerate([30, 20, 16, 14]):
        df = batch(tau, 3, "weighted", (3, ti), track=(1,))
        df["W1"] = df["Wd1"]
        for k, th in theory_stable(tau).items():
            d = describe(df[k])
            rows.append({"tau": tau, "metric": k, "theory": th, "sim_mean": d["mean"], "ci_lo": d["ci_lo"],
                         "ci_hi": d["ci_hi"], "rel_diff": (d["mean"] - th) / th if th else d["mean"],
                         "theory_in_CI": bool(d["ci_lo"] <= th <= d["ci_hi"]) if d["std"] > 0 else bool(abs(d["mean"] - th) < 1e-9)})
    df = batch(30, 3, "weighted", (3, 50), n=100, track=())
    for k, th in theory_stable(30).items():
        if k in ("lam1", "U", "busy_T4", "Pi_h"):
            d = describe(df[k])
            rows.append({"tau": 30, "metric": k + " (повтор, N=100)", "theory": th, "sim_mean": d["mean"],
                         "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "rel_diff": (d["mean"] - th) / th,
                         "theory_in_CI": bool(d["ci_lo"] <= th <= d["ci_hi"])})
    for ti, tau in enumerate([13, 12, 10, 8, 6]):
        df = batch(tau, 3, "priority", (3, 10 + ti), track=())
        th = max(0.0, 3 - 20.0 / tau)
        d = describe(df["Pi_h"])
        rows.append({"tau": tau, "metric": "Pi_h(перевант., пріоритет T2)", "theory": th, "sim_mean": d["mean"],
                     "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "rel_diff": (d["mean"] - th) / th if th else d["mean"],
                     "theory_in_CI": bool(d["ci_lo"] <= th <= d["ci_hi"])})
    for ti, tau in enumerate([13, 12, 10, 8, 6]):
        df = batch(tau, 3, "weighted", (3, 20 + ti), track=())
        th = 1.5
        d = describe(df["Pi_h"])
        rows.append({"tau": tau, "metric": "Pi_h(перевант., збалансована)", "theory": th, "sim_mean": d["mean"],
                     "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "rel_diff": (d["mean"] - th) / th,
                     "theory_in_CI": bool(d["ci_lo"] <= th <= d["ci_hi"])})
    df = batch(200, 3, "weighted", (3, 60), n=NV, ts=max(TS, 200000.0), track=(6,))
    for k, th in {"Q6": 10.0 / 200, "Wd6": 15.0, "Pi_h": 0.1}.items():
        d = describe(df[k])
        rows.append({"tau": 200, "metric": k + " (Exp-теорія, комплекти ізольовані)", "theory": th,
                     "sim_mean": d["mean"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                     "rel_diff": (d["mean"] - th) / th, "theory_in_CI": bool(d["ci_lo"] <= th <= d["ci_hi"])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "v3_theory.csv"), index=False, float_format="%.6g")
    return out


def v3_queues():
    """Черги Q2, Q3, Q6: Exp-обробка проти det; це оцінка впливу закону обробки, а не тест на збіг."""
    rows = []
    for ti, tau in enumerate([20, 14]):
        for pdist, key in (("exp", (3, 80 + ti)), ("det", (3, 90 + ti))):
            df = batch(tau, 3, "weighted", key, pdist=pdist)
            for k in ("Q2", "Q3", "Q6"):
                d = describe(df[k])
                rows.append({"tau": tau, "proc_dist": pdist, "metric": k, "mean": d["mean"], "ci_lo": d["ci_lo"],
                             "ci_hi": d["ci_hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "v3_queues_exp_vs_det.csv"), index=False, float_format="%.6g")
    return out


def v4():
    rows = []
    for ti, tau in enumerate([20, 14]):
        df = batch(tau, 3, "weighted", (4, ti))
        for k in (1, 2, 3, 6):
            a = df[f"W{k}"].to_numpy()         
            b = df[f"Wd{k}"].to_numpy()         
            m = ~np.isnan(a) & ~np.isnan(b)
            dd = describe((a - b)[m]) if m.sum() > 2 else None
            rows.append({"tau": tau, "place": f"P{k}", "W_little": np.nanmean(a), "W_direct": np.nanmean(b),
                         "mean_diff": dd["mean"] if dd else np.nan,
                         "diff_ci_lo": dd["ci_lo"] if dd else np.nan, "diff_ci_hi": dd["ci_hi"] if dd else np.nan,
                         "zero_in_CI": bool(dd and dd["ci_lo"] <= 0 <= dd["ci_hi"]),
                         "rel_diff": float((np.nanmean(a) - np.nanmean(b)) / np.nanmean(b)) if np.nanmean(b) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "v4_little.csv"), index=False, float_format="%.6g")
    return out


def v5():
    rows = []

    def add(name, df, expect, what):
        for k, ex in expect.items():
            d = describe(df[k])
            rows.append({"case": name, "metric": k, "expected": ex, "sim_mean": d["mean"], "ci_lo": d["ci_lo"],
                         "ci_hi": d["ci_hi"], "comment": what})
    add("рідкі надходження tau=200", batch(200, 3, "weighted", (5, 0), n=NV, ts=max(TS, 200000.0), track=()),
        {"Pi_h": 0.1, "Q2": 0.0, "Q3": 0.0}, "Pi = lambda = 20/tau, черг до верстатів немає (Q6=10/tau - у V3)")
    add("багато верстатів M=30, tau=14", batch(14, 30, "weighted", (5, 1), track=()),
        {"Pi_h": 20 / 14, "Q2": 0.0, "Q3": 0.0}, "обмеження верстатами знято: Pi = lambda, Q2=Q3=0")
    add("дуже часті tau=8, збалансована b=1/3:2/3", batch(8, 3, "weighted", (5, 2), track=()),
        {"Pi_h": 1.5}, "Pi -> Pi_max = M/2 = 1,5 виробу/год")
    add("дуже часті tau=6, пріоритет T2", batch(6, 3, "priority", (5, 3), track=()),
        {"Pi_h": 0.0}, "верстати цілком зайняті деталями типу 1: Pi -> 0 (голодування типу 2)")
    add("tau=20, M=1 (нестійка)", batch(20, 1, "weighted", (5, 4), track=()),
        {"U": 1.0}, "B=2 > M=1: верстат завантажений на 100%")
    out = pd.DataFrame(rows)
    out["expected_in_CI_or_close"] = [(r.ci_lo <= r.expected <= r.ci_hi) or abs(r.sim_mean - r.expected) <= 0.03 * max(1.0, abs(r.expected))
                                      for r in out.itertuples()]
    out.to_csv(os.path.join(RES, "v5_boundary.csv"), index=False, float_format="%.6g")
    return out


if __name__ == "__main__":
    summ = {}
    ok1, det = v1()
    summ["V1"] = {"ok": bool(ok1), "n_iterations_compared": len(EXPECTED), "long_deterministic_run": det}
    print("V1", ok1, det)
    ok2, o2 = v2()
    summ["V2"] = {"ok": bool(ok2), "total_iterations_checked": o2["total_iterations_checked"],
                  "tamper_detected": o2["tamper_detected"]}
    print("V2", ok2, o2["total_iterations_checked"], o2["tamper_detected"])
    t3 = v3()
    pd.set_option("display.width", 220)
    print(t3.to_string())
    summ["V3"] = {"n": len(t3), "n_theory_in_CI": int(t3.theory_in_CI.sum()),
                  "max_abs_rel_diff": float(t3.rel_diff.abs().max())}
    t3q = v3_queues()
    print(t3q.to_string())
    t4 = v4()
    print(t4.to_string())
    summ["V3_queues_exp_vs_det"] = t3q.to_dict("records")
    summ["V4"] = {"n": len(t4), "n_zero_in_CI": int(t4.zero_in_CI.sum()),
                  "max_abs_rel_diff": float(t4.rel_diff.abs().max())}
    t5 = v5()
    print(t5.to_string())
    summ["V5"] = {"n": len(t5), "n_ok": int(t5.expected_in_CI_or_close.sum())}
    summ["params"] = {"T_warm": TW, "T_stat": TS, "N": NV}
    json.dump(summ, open(os.path.join(RES, "verification_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(summ)
