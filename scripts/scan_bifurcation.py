import os, json
import numpy as np

from src.utils import read_json, write_json, make_run_dir
from src.model import build_params, build_noise, step_em
from src.plotting import save_bifurcation

def extract_local_maxima(t, x):
    # simple maxima: x[n-1] < x[n] > x[n+1]
    m = []
    for i in range(1, len(x)-1):
        if x[i-1] < x[i] and x[i] > x[i+1]:
            m.append(float(x[i]))
    return m

def run_for_param(cfg, param_name, param_value):
    # modify cfg in-place for one run
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["model"][param_name] = float(param_value)

    nx = int(cfg2["grid"]["nx"]); ny = int(cfg2["grid"]["ny"])
    boundary = str(cfg2["grid"]["boundary"])
    Kx = float(cfg2["coupling"]["Kx"])

    p = build_params(cfg2)
    noise = build_noise(cfg2)

    dt = float(cfg2["sim"]["dt"])
    t_total = float(cfg2["sim"]["t_total"])
    burn_in = float(cfg2["sim"]["burn_in"])
    sample_every = int(cfg2["sim"]["sample_every"])
    seed = int(cfg2["sim"]["seed"])

    trace_i, trace_j = cfg2["analysis"]["trace_node"]
    trace_i = int(trace_i); trace_j = int(trace_j)

    rng = np.random.default_rng(seed)
    x = 0.1 + 0.05 * rng.standard_normal((nx, ny))
    y = 0.3 + 0.05 * rng.standard_normal((nx, ny))
    E = 0.003 + 0.001 * rng.standard_normal((nx, ny))

    steps = int(t_total / dt)
    t = 0.0
    tt = []
    xx = []
    for n in range(steps):
        x, y, E = step_em(x, y, E, t, dt, p, Kx, boundary, noise, rng)
        t += dt
        if n % sample_every == 0:
            tt.append(t)
            xx.append(float(x[trace_i, trace_j]))
    tt = np.array(tt, dtype=float)
    xx = np.array(xx, dtype=float)

    mask = tt >= burn_in
    maxima = extract_local_maxima(tt[mask], xx[mask])
    return maxima

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--param", required=True, choices=["omega","f","Em","T","b","Kx"])
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--num", type=int, default=60)
    args = ap.parse_args()

    cfg = read_json(args.config)
    run_dir = make_run_dir("runs", {**cfg, "scan": vars(args)})

    # allow scanning Kx which lives in coupling
    pvals = np.linspace(args.start, args.stop, args.num)
    maxima_list = []

    for pv in pvals:
        if args.param == "Kx":
            cfg_mod = json.loads(json.dumps(cfg))
            cfg_mod["coupling"]["Kx"] = float(pv)
            maxima = run_for_param(cfg_mod, "omega", cfg_mod["model"]["omega"])  # omega unchanged
        else:
            maxima = run_for_param(cfg, args.param, pv)
        maxima_list.append(maxima)

    write_json(os.path.join(run_dir, "scan_args.json"), vars(args))
    write_json(os.path.join(run_dir, "config_used.json"), cfg)

    # Save raw
    np.savez_compressed(os.path.join(run_dir, "bifurcation_data.npz"),
                        pvals=pvals, maxima=np.array([np.array(m, dtype=float) for m in maxima_list], dtype=object))

    save_bifurcation(
        os.path.join(run_dir, "bifurcation.png"),
        pvals, maxima_list,
        xlabel=args.param,
        title=f"Bifurcation-style diagram (node {cfg['analysis']['trace_node']})"
    )

    print("SCAN_DIR:", run_dir)

if __name__ == "__main__":
    main()
