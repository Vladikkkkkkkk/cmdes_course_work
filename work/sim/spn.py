import heapq
from collections import deque


class Transition:

    def __init__(self, name, ins, outs, delay=0.0, r=0, b=1.0):
        self.name, self.ins, self.outs = name, list(ins), list(outs)
        self.r, self.b = r, b
        self.delay_spec = delay
        if callable(delay):
            self.sample = delay
        elif isinstance(delay, tuple) and delay[0] == "unif":
            lo, hi = float(delay[1]), float(delay[2])
            self.sample = lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi)
        elif isinstance(delay, tuple) and delay[0] == "exp":
            mean = float(delay[1])     
            self.sample = lambda rng, m=mean: rng.exponential(m)
        else:
            d = float(delay)
            self.sample = lambda rng, d=d: d


class Net:
    def __init__(self, places, mu0, transitions):
        assert len(places) == len(mu0)
        self.places, self.mu0, self.trs = list(places), list(mu0), list(transitions)


class Simulator:
    EPS = 1e-12

    def __init__(self, net, rng, track=(), keep_samples=False, check=None, trace=None):
        self.net, self.rng = net, rng
        self.mu = list(net.mu0)
        nP, nT = len(self.mu), len(net.trs)
        self.nP, self.nT = nP, nT
        self.t = 0.0
        self.cal = []          
        self._seq = 0
        self.act = [0] * nT    
        self.starts = [0] * nT  
        self.ends = [0] * nT
        self.n_iter = 0
        self.track = {p: deque([0.0] * self.mu[p]) for p in track}
        self.keep = keep_samples
        self.check, self.trace = check, trace
        self.reset_stats(0.0)
        self._fire_loop()      
        self._after_iter()

    # ---------------- статистика ----------------
    def reset_stats(self, t):
        self.last = t
        self.t_stat0 = t
        self.integ = [0.0] * self.nP
        self.busy = [0.0] * self.nT
        self.maxmu = list(self.mu)
        self.entries = [0] * self.nP
        self.st_starts = [0] * self.nT
        self.st_ends = [0] * self.nT
        self.wsum = {p: 0.0 for p in self.track}
        self.wn = {p: 0 for p in self.track}
        self.wmax = {p: 0.0 for p in self.track}
        self.samples = {p: [] for p in self.track}

    def _advance(self, t):
        dt = t - self.last
        if dt > 0:
            mu, integ, act, busy = self.mu, self.integ, self.act, self.busy
            for p in range(self.nP):
                integ[p] += mu[p] * dt
            for i in range(self.nT):
                busy[i] += act[i] * dt
        self.last = t

    # ---------------- ядро ----------------
    def _enabled(self):
        mu = self.mu
        return [i for i, tr in enumerate(self.net.trs)
                if all(mu[p] >= k for p, k in tr.ins)]

    def _resolve(self, enabled):
        """Вирішення конфлікту: найвищий пріоритет r; у групі - вибір за вагами b."""
        trs = self.net.trs
        rmax = max(trs[i].r for i in enabled)
        group = [i for i in enabled if trs[i].r == rmax]
        if len(group) == 1:
            return group[0]  
        w = [trs[i].b for i in group]
        s = sum(w)
        u = self.rng.random() * s
        acc = 0.0
        for i, wi in zip(group, w):
            acc += wi
            if u < acc:
                return i
        return group[-1]

    def _fire(self, i):
        tr, t, mu = self.net.trs[i], self.t, self.mu
        for p, k in tr.ins:
            mu[p] -= k
            dq = self.track.get(p)
            if dq is not None:
                for _ in range(k):
                    w = t - dq.popleft()
                    self.wsum[p] += w
                    self.wn[p] += 1
                    if w > self.wmax[p]:
                        self.wmax[p] = w
                    if self.keep:
                        self.samples[p].append(w)
        self.act[i] += 1
        self.starts[i] += 1
        self.st_starts[i] += 1
        self._seq += 1
        heapq.heappush(self.cal, (t + tr.sample(self.rng), self._seq, i))

    def _fire_loop(self):
        while True:
            en = self._enabled()
            if not en:
                break
            self._fire(self._resolve(en))

    def _output(self, i):
        tr, t, mu = self.net.trs[i], self.t, self.mu
        for p, k in tr.outs:
            mu[p] += k
            self.entries[p] += k
            dq = self.track.get(p)
            if dq is not None:
                dq.extend([t] * k)
        self.act[i] -= 1
        self.ends[i] += 1
        self.st_ends[i] += 1

    def _after_iter(self):
        self.n_iter += 1
        mu, mx = self.mu, self.maxmu
        for p in range(self.nP):
            if mu[p] > mx[p]:
                mx[p] = mu[p]
        if self.check is not None:
            self.check(self)
        if self.trace is not None:
            self.trace.append((self.t, tuple(mu), tuple(self.act)))

    def step(self):
        t = self.cal[0][0]
        self._advance(t)
        self.t = t
        while self.cal and self.cal[0][0] <= t + self.EPS:
            _, _, i = heapq.heappop(self.cal)
            self._output(i)
        self._fire_loop()
        self._after_iter()

    def run(self, until):
        while self.cal and self.cal[0][0] <= until:
            self.step()
        self._advance(until)
        self.t = until


def check_balance(sim):
    net = sim.net
    bal = list(net.mu0)
    for i, tr in enumerate(net.trs):
        assert sim.act[i] == sim.starts[i] - sim.ends[i], f"канали {tr.name}"
        for p, k in tr.ins:
            bal[p] -= k * sim.starts[i]
        for p, k in tr.outs:
            bal[p] += k * sim.ends[i]
    assert bal == sim.mu, f"баланс маркерів порушено t={sim.t}: {bal} != {sim.mu}"
