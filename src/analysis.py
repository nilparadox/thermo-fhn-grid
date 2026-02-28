import numpy as np
from typing import Dict, Any, List, Tuple

def detect_spikes(t: np.ndarray, x: np.ndarray, thr: float, min_sep: float) -> List[float]:
    # spikes when x crosses threshold upward
    # x: (T,)
    above = x >= thr
    crossings = np.where((~above[:-1]) & (above[1:]))[0] + 1
    spike_times = []
    last = -1e18
    for idx in crossings:
        ts = float(t[idx])
        if ts - last >= min_sep:
            spike_times.append(ts)
            last = ts
    return spike_times

def isi_cv(spike_times: List[float]) -> float:
    if len(spike_times) < 3:
        return float("nan")
    isi = np.diff(spike_times)
    m = float(np.mean(isi))
    if m <= 0:
        return float("nan")
    v = float(np.std(isi))
    return v / m

def network_cv(t: np.ndarray, X: np.ndarray, thr: float, min_sep: float, burn_in: float) -> Dict[str, float]:
    # X: (T, nx, ny)
    mask = t >= burn_in
    tt = t[mask]
    XX = X[mask]
    cvs = []
    nx, ny = XX.shape[1], XX.shape[2]
    for i in range(nx):
        for j in range(ny):
            spikes = detect_spikes(tt, XX[:, i, j], thr, min_sep)
            cv = isi_cv(spikes)
            if np.isfinite(cv):
                cvs.append(cv)
    if len(cvs) == 0:
        return {"cv_mean": float("nan"), "cv_median": float("nan"), "cv_n": 0}
    cvs = np.array(cvs, dtype=float)
    return {"cv_mean": float(np.mean(cvs)), "cv_median": float(np.median(cvs)), "cv_n": int(len(cvs))}

def phase_from_spikes(t: np.ndarray, x: np.ndarray, thr: float) -> np.ndarray:
    # crude phase from spike times: linear phase between spikes
    above = x >= thr
    crossings = np.where((~above[:-1]) & (above[1:]))[0] + 1
    if len(crossings) < 2:
        return np.zeros_like(t)
    st = t[crossings]
    phi = np.zeros_like(t, dtype=float)
    # for each interval [st[k], st[k+1]) phase goes 0->2pi
    for k in range(len(st) - 1):
        t0, t1 = st[k], st[k+1]
        m = (t >= t0) & (t < t1)
        if t1 > t0:
            phi[m] = 2.0 * np.pi * (t[m] - t0) / (t1 - t0)
    # after last spike, hold phase
    m2 = t >= st[-1]
    phi[m2] = 0.0
    return phi

def kuramoto_R(t: np.ndarray, X: np.ndarray, thr: float, burn_in: float) -> float:
    # compute mean phase coherence R over time after burn-in
    mask = t >= burn_in
    tt = t[mask]
    XX = X[mask]
    nx, ny = XX.shape[1], XX.shape[2]
    Rts = []
    # compute phase for each node (expensive but ok for 625)
    phases = np.zeros((len(tt), nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            phases[:, i, j] = phase_from_spikes(tt, XX[:, i, j], thr)
    # R(t)
    z = np.exp(1j * phases.reshape(len(tt), nx * ny))
    Rt = np.abs(np.mean(z, axis=1))
    return float(np.mean(Rt))

def lyapunov_benettin(stepper, state0, dt: float, t0: float, total_time: float,
                      renorm_dt: float, eps0: float = 1e-7) -> float:
    """
    Largest Lyapunov exponent using 1 tangent vector (Benettin).
    stepper: function(state, t, dt) -> next_state
    state0: tuple of arrays (x,y,E)
    """
    x0, y0, E0 = state0
    rng = np.random.default_rng(0)
    # random perturbation
    dx = rng.standard_normal(size=x0.shape)
    dy = rng.standard_normal(size=y0.shape)
    dE = rng.standard_normal(size=E0.shape)

    # normalize
    norm = np.sqrt(np.sum(dx*dx) + np.sum(dy*dy) + np.sum(dE*dE))
    dx = dx / norm * eps0
    dy = dy / norm * eps0
    dE = dE / norm * eps0

    x1 = x0 + dx
    y1 = y0 + dy
    E1 = E0 + dE

    t = t0
    steps = int(total_time / dt)
    ren_steps = max(1, int(renorm_dt / dt))
    sum_log = 0.0
    m = 0

    for n in range(steps):
        x0, y0, E0 = stepper((x0, y0, E0), t, dt)
        x1, y1, E1 = stepper((x1, y1, E1), t, dt)
        t += dt

        if (n + 1) % ren_steps == 0:
            ddx = x1 - x0
            ddy = y1 - y0
            ddE = E1 - E0
            dist = np.sqrt(np.sum(ddx*ddx) + np.sum(ddy*ddy) + np.sum(ddE*ddE))
            if dist <= 0:
                continue
            sum_log += np.log(dist / eps0)
            m += 1
            # renormalize
            scale = eps0 / dist
            x1 = x0 + ddx * scale
            y1 = y0 + ddy * scale
            E1 = E0 + ddE * scale

    if m == 0:
        return float("nan")
    return float(sum_log / (m * renorm_dt))
