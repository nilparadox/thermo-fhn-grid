import os, json
import numpy as np

from src.utils import read_json, write_json, make_run_dir
from src.model import build_params, build_noise, step_em
from src.analysis import network_cv, kuramoto_R, lyapunov_benettin
from src.plotting import save_timeseries_plot

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    args = ap.parse_args()

    cfg = read_json(args.config)
    run_dir = make_run_dir("runs", cfg)
    write_json(os.path.join(run_dir, "config_used.json"), cfg)

    nx = int(cfg["grid"]["nx"]); ny = int(cfg["grid"]["ny"])
    boundary = str(cfg["grid"]["boundary"])
    Kx = float(cfg["coupling"]["Kx"])

    p = build_params(cfg)
    noise = build_noise(cfg)

    dt = float(cfg["sim"]["dt"])
    t_total = float(cfg["sim"]["t_total"])
    burn_in = float(cfg["sim"]["burn_in"])
    sample_every = int(cfg["sim"]["sample_every"])
    seed = int(cfg["sim"]["seed"])

    thr = float(cfg["analysis"]["spike_threshold"])
    min_sep = float(cfg["analysis"]["min_spike_separation"])

    rng = np.random.default_rng(seed)

    # init (small random)
    x = 0.1 + 0.05 * rng.standard_normal((nx, ny))
    y = 0.3 + 0.05 * rng.standard_normal((nx, ny))
    E = 0.003 + 0.001 * rng.standard_normal((nx, ny))

    steps = int(t_total / dt)
    t = 0.0

    # storage
    ts_t = []
    ts_xc = []
    # representative node for plotting
    trace_i, trace_j = cfg["analysis"]["trace_node"]
    trace_i = int(trace_i); trace_j = int(trace_j)

    for n in range(steps):
        x, y, E = step_em(x, y, E, t, dt, p, Kx, boundary, noise, rng)
        t += dt
        if n % sample_every == 0:
            ts_t.append(t)
            ts_xc.append(float(x[trace_i, trace_j]))

    ts_t = np.array(ts_t, dtype=float)
    ts_xc = np.array(ts_xc, dtype=float)

    # For network-level stats, store x snapshots at sample points (compressed)
    # Keep memory reasonable: store every sample_every step only.
    # Re-simulate for full X history at sample points (deterministic given same seed).
    rng2 = np.random.default_rng(seed)
    x2 = 0.1 + 0.05 * rng2.standard_normal((nx, ny))
    y2 = 0.3 + 0.05 * rng2.standard_normal((nx, ny))
    E2 = 0.003 + 0.001 * rng2.standard_normal((nx, ny))
    t2 = 0.0
    Xhist = []
    Thist = []
    for n in range(steps):
        x2, y2, E2 = step_em(x2, y2, E2, t2, dt, p, Kx, boundary, noise, rng2)
        t2 += dt
        if n % sample_every == 0:
            Thist.append(t2)
            Xhist.append(x2.copy())
    Thist = np.array(Thist, dtype=float)
    Xhist = np.stack(Xhist, axis=0)  # (T, nx, ny)

    np.savez_compressed(os.path.join(run_dir, "timeseries.npz"),
                        t=Thist, X=Xhist,
                        trace_t=ts_t, trace_x=ts_xc)

    # Metrics: CV + coherence
    cv_stats = network_cv(Thist, Xhist, thr=thr, min_sep=min_sep, burn_in=burn_in)
    R = kuramoto_R(Thist, Xhist, thr=thr, burn_in=burn_in)

    metrics = {
        "cv_mean": cv_stats["cv_mean"],
        "cv_median": cv_stats["cv_median"],
        "cv_n": cv_stats["cv_n"],
        "kuramoto_R": R
    }

    # Largest Lyapunov exponent (optional)
    if bool(cfg["analysis"]["compute_lyapunov"]):
        lyap_total = float(cfg["analysis"]["lyap_total_time"])
        ren_dt = float(cfg["analysis"]["lyap_renorm_dt"])

        # Build a deterministic stepper closure with SAME noise stream:
        # For Lyapunov, we need consistent stochastic forcing for both trajectories
        # (common-noise Lyapunov). We implement stepper with a pre-generated noise tape.
        # This is defendable: common-noise LE measures divergence from initial condition under identical forcing.
        steps_ly = int(lyap_total / dt)
        rngL = np.random.default_rng(seed + 999)
        if noise.enable:
            tape_x = rngL.standard_normal((steps_ly, nx, ny))
            tape_y = rngL.standard_normal((steps_ly, nx, ny))
            tape_E = rngL.standard_normal((steps_ly, nx, ny))
        else:
            tape_x = tape_y = tape_E = None

        # initial state for Lyap: use last state after burn-in by re-sim
        # so LE reflects steady regime
        rng3 = np.random.default_rng(seed)
        x3 = 0.1 + 0.05 * rng3.standard_normal((nx, ny))
        y3 = 0.3 + 0.05 * rng3.standard_normal((nx, ny))
        E3 = 0.003 + 0.001 * rng3.standard_normal((nx, ny))
        t3 = 0.0
        burn_steps = int(burn_in / dt)
        for n in range(burn_steps):
            x3, y3, E3 = step_em(x3, y3, E3, t3, dt, p, Kx, boundary, noise, rng3)
            t3 += dt

        # Now define deterministic step using tape for EM
        idx = {"n": 0}
        def stepper(state, tcur, dtcur):
            xS, yS, ES = state
            # compute drift
            from src.model import drift, e_ext
            dx, dy, dE = drift(xS, yS, ES, tcur, p, Kx, boundary)
            if noise.enable:
                n = idx["n"]
                xS = xS + dx * dtcur + noise.sigma_x * np.sqrt(dtcur) * tape_x[n]
                yS = yS + dy * dtcur + noise.sigma_y * np.sqrt(dtcur) * tape_y[n]
                ES = ES + dE * dtcur + noise.sigma_E * np.sqrt(dtcur) * tape_E[n]
                idx["n"] = n + 1
            else:
                xS = xS + dx * dtcur
                yS = yS + dy * dtcur
                ES = ES + dE * dtcur
            return (xS, yS, ES)

        # Important: Benettin calls stepper twice per step; reset tape index inside wrapper
        # We'll provide a wrapper that advances both trajectories with same tape values.
        # Implement as: at each n, advance both using same tape[n], then idx increments once.
        idx2 = {"n": 0}
        def stepper_common(state, tcur, dtcur):
            xS, yS, ES = state
            from src.model import drift
            dx, dy, dE = drift(xS, yS, ES, tcur, p, Kx, boundary)
            if noise.enable:
                n = idx2["n"]
                xS = xS + dx * dtcur + noise.sigma_x * np.sqrt(dtcur) * tape_x[n]
                yS = yS + dy * dtcur + noise.sigma_y * np.sqrt(dtcur) * tape_y[n]
                ES = ES + dE * dtcur + noise.sigma_E * np.sqrt(dtcur) * tape_E[n]
            else:
                xS = xS + dx * dtcur
                yS = yS + dy * dtcur
                ES = ES + dE * dtcur
            return (xS, yS, ES)

        # Modified Benettin loop with common noise:
        def lyap_common_noise(state0):
            x0L, y0L, E0L = state0
            rngP = np.random.default_rng(0)
            dx0 = rngP.standard_normal(size=x0L.shape)
            dy0 = rngP.standard_normal(size=y0L.shape)
            dE0 = rngP.standard_normal(size=E0L.shape)
            norm = np.sqrt(np.sum(dx0*dx0)+np.sum(dy0*dy0)+np.sum(dE0*dE0))
            eps0 = 1e-7
            dx0 = dx0/norm*eps0; dy0 = dy0/norm*eps0; dE0 = dE0/norm*eps0
            x1L = x0L + dx0; y1L = y0L + dy0; E1L = E0L + dE0

            tcur = t3
            ren_steps = max(1, int(ren_dt / dt))
            sum_log = 0.0
            mcount = 0

            for n in range(steps_ly):
                # use same tape[n] for both
                idx2["n"] = n
                x0L, y0L, E0L = stepper_common((x0L, y0L, E0L), tcur, dt)
                idx2["n"] = n
                x1L, y1L, E1L = stepper_common((x1L, y1L, E1L), tcur, dt)
                tcur += dt

                if (n+1) % ren_steps == 0:
                    ddx = x1L - x0L
                    ddy = y1L - y0L
                    ddE = E1L - E0L
                    dist = np.sqrt(np.sum(ddx*ddx)+np.sum(ddy*ddy)+np.sum(ddE*ddE))
                    if dist > 0:
                        sum_log += np.log(dist / eps0)
                        mcount += 1
                        scale = eps0 / dist
                        x1L = x0L + ddx * scale
                        y1L = y0L + ddy * scale
                        E1L = E0L + ddE * scale

            if mcount == 0:
                return float("nan")
            return float(sum_log / (mcount * ren_dt))

        lam = lyap_common_noise((x3, y3, E3))
        metrics["lyap_max_common_noise"] = lam

    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    # Plot representative node
    save_timeseries_plot(
        os.path.join(run_dir, "trace_x.png"),
        Thist, Xhist[:, trace_i, trace_j],
        title=f"Trace x(t) at node ({trace_i},{trace_j})"
    )

    print("RUN_DIR:", run_dir)
    print("METRICS:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
