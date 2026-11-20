import numpy as np
from spn import Net, Transition, Simulator, check_balance

PLACES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
TRANS = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]


def build_net(tau=20.0, M=3, policy="weighted", width=5.0, deterministic=False, proc_dist="exp"):
    if policy == "priority":      
        r2, r3, b2, b3 = 1, 0, 1.0, 1.0
    elif policy == "priority3":   
        r2, r3, b2, b3 = 0, 1, 1.0, 1.0
    elif policy == "random":      
        r2, r3, b2, b3 = 0, 0, 0.5, 0.5
    elif policy == "weighted":    
        r2, r3, b2, b3 = 0, 0, 1.0 / 3.0, 2.0 / 3.0
    else:
        raise ValueError(policy)
    if proc_dist not in ("exp", "det"):
        raise ValueError(proc_dist)
    if deterministic:
        proc_dist = "det"
    d2, d3 = (("exp", 60.0), ("exp", 30.0)) if proc_dist == "exp" else (60.0, 30.0)
    d0 = tau if deterministic else ("unif", tau - width, tau + width)   
    d4 = 60.0 if deterministic else ("unif", 58.0, 62.0)
    d5 = 60.0 if deterministic else ("unif", 52.0, 68.0)
    trs = [
        Transition("T0", [(0, 1)], [(0, 1), (1, 1)], d0),
        Transition("T1", [(1, 3)], [(2, 1), (3, 2)], 0.0),
        Transition("T2", [(2, 1), (4, 1)], [(4, 1), (5, 1)], d2, r=r2, b=b2),
        Transition("T3", [(3, 1), (4, 1)], [(4, 1), (6, 1)], d3, r=r3, b=b3),
        Transition("T4", [(5, 1)], [(7, 1)], d4),
        Transition("T5", [(6, 2)], [(8, 1)], d5),
        Transition("T6", [(7, 1), (8, 1)], [(9, 1)], 0.0),
    ]
    return Net(PLACES, [1, 0, 0, 0, M, 0, 0, 0, 0, 0], trs)


def make_checker(M):
    def check(s):
        check_balance(s)
        mu, a, st, en = s.mu, s.act, s.starts, s.ends
        assert mu[0] + a[0] == 1, "I1: P0 + T0 = 1"
        assert mu[4] + a[2] + a[3] == M, "I2: ресурс верстатів"
        assert en[0] == mu[1] + 3 * st[1], "I3: заготовки"
        assert en[1] == mu[2] + st[2], "I4a"
        assert 2 * en[1] == mu[3] + st[3], "I4b"
        assert en[2] == mu[5] + st[4], "I5a"
        assert en[3] == mu[6] + 2 * st[5], "I5b"
        assert en[4] == mu[7] + st[6], "I6a"
        assert en[5] == mu[8] + st[6], "I6b"
        assert en[6] == mu[9], "I7"
    return check


def collect(sim, T):
    I, E = sim.integ, sim.entries
    M = sim.net.mu0[4]
    out = {"T": T, "N_prod": E[9]}
    out["Pi_h"] = E[9] / T * 60.0                       
    out["U"] = 1.0 - I[4] / (M * T)
    out["B"] = M * out["U"]
    for k in (1, 2, 3, 5, 6, 7, 8):
        out[f"Q{k}"] = I[k] / T
        out[f"max{k}"] = sim.maxmu[k]
        out[f"lam{k}"] = E[k] / T
    out["Q23"] = out["Q2"] + out["Q3"]
    for k in (1, 2, 3, 6):                              
        out[f"W{k}"] = out[f"Q{k}"] / out[f"lam{k}"] if out[f"lam{k}"] > 0 else float("nan")
    for p in sim.track:                                
        out[f"Wd{p}"] = sim.wsum[p] / sim.wn[p] if sim.wn[p] else float("nan")
        out[f"Wmax{p}"] = sim.wmax[p]
    out["busy_T4"] = sim.busy[4] / T
    out["busy_T5"] = sim.busy[5] / T
    out["busy_T2"] = sim.busy[2] / T
    out["busy_T3"] = sim.busy[3] / T
    out["sets"] = sim.st_starts[1]
    return out


def run_model(tau, M, policy, rng, t_warm, t_stat, track=(1, 2, 3, 5, 6), check=False,
              keep_samples=False, width=5.0, proc_dist="exp"):
    net = build_net(tau, M, policy, width, proc_dist=proc_dist)
    sim = Simulator(net, rng, track=track, keep_samples=keep_samples,
                    check=make_checker(M) if check else None)
    sim.run(t_warm)
    sim.reset_stats(t_warm)
    sim.run(t_warm + t_stat)
    res = collect(sim, t_stat)
    res["_sim"] = sim if keep_samples else None
    return res


def run_series(tau, M, policy, rng, horizon, w, width=5.0, proc_dist="exp"):
    sim = Simulator(build_net(tau, M, policy, width, proc_dist=proc_dist), rng, track=())
    n = int(round(horizon / w))
    rows = []
    prev_I, prev_E = list(sim.integ), list(sim.entries)
    for j in range(1, n + 1):
        sim.run(j * w)
        I, E = sim.integ, sim.entries
        dI = [a - b for a, b in zip(I, prev_I)]
        dE = [a - b for a, b in zip(E, prev_E)]
        rows.append((dE[9] * 60.0 / w, (dI[2] + dI[3]) / w, dI[1] / w,
                     sum(dI[p] for p in (1, 2, 3, 5, 6, 7, 8)) / w))
        prev_I, prev_E = list(I), list(E)
    return np.array(rows)
