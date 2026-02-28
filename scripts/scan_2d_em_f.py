import os, json
import numpy as np

from src.utils import read_json, write_json, make_run_dir
from src.model import build_params, build_noise, step_em
from src.analysis import network_cv, kuramoto_R
from src.plotting import save_heatmap

def simulate_metrics(cfg):
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

    x = 0.1 + 0.05 * rng.standard_normal((nx, ny))
    y = 0.3 + 0.05 * rng.standard_normal((nx, ny))
    E = 0.003 + 0.001 * rng.standard_normal((nx, ny))

    steps = int(t_total / dt)
    t = 0.0
    Thist = []
    Xhist = []

    for n in range(steps):
        x, y, E = step_em(x, y, E, t, dt, p, Kx, boundary, noise, rng)
        t += dt
        if n % sample_every == 0:
            Thist.append(t)
            Xhist.append(x.copy())

    Thist = np.array(Thist, dtype=float)
    Xhist = np.stack(Xhist, axis=0)  # (T, nx, ny)

    cv_stats = network_cv(Thist, Xhist, thr=thr, min_sep=min_sep, burn_in=burn_in)
    R = kuramoto_R(Thist, Xhist, thr=thr, burn_in=burn_in)
    return cv_stats["cv_mean"], R

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--Em_min", type=float, default=0.0)
    ap.add_argument("--Em_max", type=float, default=3.0)
    ap.add_argument("--Em_num", type=int, default=21)
    ap.add_argument("--f_min", type=float, default=0.0)
    ap.add_argument("--f_max", type=float, default=0.05)
    ap.add_argument("--f_num", type=int, default=21)
    args = ap.parse_args()

    cfg = read_json(args.config)
    scan_cfg = {**cfg, "scan": vars(args)}
    run_dir = make_run_dir("runs", scan_cfg)
    write_json(os.path.join(run_dir, "config_used.json"), cfg)
    write_json(os.path.join(run_dir, "scan_args.json"), vars(args))

    Em_vals = np.linspace(args.Em_min, args.Em_max, args.Em_num)
    f_vals  = np.linspace(args.f_min,  args.f_max,  args.f_num)

    CV = np.full((len(f_vals), len(Em_vals)), np.nan, dtype=float)
    RR = np.full((len(f_vals), len(Em_vals)), np.nan, dtype=float)

    for iy, fv in enumerate(f_vals):
        for ix, Em in enumerate(Em_vals):
            cfg2 = json.loads(json.dumps(cfg))
            cfg2["model"]["Em"] = float(Em)
            cfg2["model"]["f"] = float(fv)

            cvm, R = simulate_metrics(cfg2)
            CV[iy, ix] = cvm
            RR[iy, ix] = R

            print(f"[{iy+1}/{len(f_vals)}][{ix+1}/{len(Em_vals)}] f={fv:.5f} Em={Em:.3f}  CV={cvm:.4f}  R={R:.4f}")

    np.savez_compressed(os.path.join(run_dir, "scan_em_f.npz"),
                        Em_vals=Em_vals, f_vals=f_vals, CV=CV, R=RR)

    save_heatmap(os.path.join(run_dir, "heatmap_CV.png"),
                 CV, Em_vals, f_vals,
                 xlabel="Em", ylabel="f",
                 title="Network CV mean (spike irregularity)")

    save_heatmap(os.path.join(run_dir, "heatmap_R.png"),
                 RR, Em_vals, f_vals,
                 xlabel="Em", ylabel="f",
                 title="Kuramoto R (phase coherence)")

    print("SCAN_DIR:", run_dir)

if __name__ == "__main__":
    main()
