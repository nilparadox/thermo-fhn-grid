import os, json, csv
import numpy as np

from src.utils import read_json, write_json, make_run_dir
from src.model import build_params, build_noise, step_em
from src.analysis import network_cv, kuramoto_R, detect_spikes
from src.plotting import save_heatmap, save_timeseries_plot

import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def deep_copy_cfg(cfg):
    return json.loads(json.dumps(cfg))

def simulate_store_X(cfg):
    """
    Simulate and store X(t) at sampled points.
    Returns Thist (T,), Xhist (T,nx,ny)
    """
    nx = int(cfg["grid"]["nx"]); ny = int(cfg["grid"]["ny"])
    boundary = str(cfg["grid"]["boundary"])
    Kx = float(cfg["coupling"]["Kx"])

    p = build_params(cfg)
    noise = build_noise(cfg)

    dt = float(cfg["sim"]["dt"])
    t_total = float(cfg["sim"]["t_total"])
    sample_every = int(cfg["sim"]["sample_every"])
    seed = int(cfg["sim"]["seed"])

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
    Xhist = np.stack(Xhist, axis=0)
    return Thist, Xhist

def compute_metrics(cfg, Thist, Xhist):
    burn_in = float(cfg["sim"]["burn_in"])
    thr = float(cfg["analysis"]["spike_threshold"])
    min_sep = float(cfg["analysis"]["min_spike_separation"])
    cv_stats = network_cv(Thist, Xhist, thr=thr, min_sep=min_sep, burn_in=burn_in)
    R = kuramoto_R(Thist, Xhist, thr=thr, burn_in=burn_in)
    return {
        "cv_mean": cv_stats["cv_mean"],
        "cv_median": cv_stats["cv_median"],
        "cv_n": cv_stats["cv_n"],
        "kuramoto_R": R
    }

def save_lineplot(out_png, x, y_list, labels, xlabel, ylabel, title):
    ensure_dir(os.path.dirname(out_png))
    plt.figure()
    for y, lab in zip(y_list, labels):
        plt.plot(x, y, marker="o", label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_twoaxis(out_png, x, y1, y2, xlabel, y1label, y2label, title):
    ensure_dir(os.path.dirname(out_png))
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, marker="o")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, marker="s")
    ax2.set_ylabel(y2label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def raster_from_spikes(out_png, Thist, Xhist, thr, min_sep, burn_in, max_neurons=400):
    """
    Spike raster plot: pick up to max_neurons nodes (uniform subsample) for readability.
    """
    ensure_dir(os.path.dirname(out_png))
    t = Thist
    mask = t >= burn_in
    tt = t[mask]
    XX = Xhist[mask]

    nx, ny = XX.shape[1], XX.shape[2]
    N = nx * ny

    # subsample indices
    if N <= max_neurons:
        indices = np.arange(N)
    else:
        indices = np.linspace(0, N-1, max_neurons).astype(int)

    spike_t = []
    spike_y = []

    for row, idx in enumerate(indices):
        i = idx // ny
        j = idx % ny
        spikes = detect_spikes(tt, XX[:, i, j], thr=thr, min_sep=min_sep)
        for s in spikes:
            spike_t.append(s)
            spike_y.append(row)

    plt.figure(figsize=(8,4))
    plt.plot(spike_t, spike_y, "k.", markersize=1.5)
    plt.xlabel("t")
    plt.ylabel("neuron (subsample index)")
    plt.title("Spike raster (subsampled neurons)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------------
# Main pipeline
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="grid_network_outputs")
    args = ap.parse_args()

    cfg = read_json(args.config)

    # Create ONE run folder for everything
    run_cfg = deep_copy_cfg(cfg)
    run_cfg["run_label"] = args.label
    run_dir = make_run_dir("runs", run_cfg)

    # folder layout
    out_maps = os.path.join(run_dir, "maps_em_f")
    out_slices = os.path.join(run_dir, "slices_frequency")
    out_temp = os.path.join(run_dir, "temperature_sweep")
    out_examples = os.path.join(run_dir, "examples")
    out_raw = os.path.join(run_dir, "raw")
    for d in [out_maps, out_slices, out_temp, out_examples, out_raw]:
        ensure_dir(d)

    # Save config
    write_json(os.path.join(run_dir, "config_used.json"), cfg)

    # -------------------------
    # 1) Phase map: Em x f -> (CV, R)
    # -------------------------
    scan = cfg.get("paper_outputs", {}).get("em_f_map", {})
    Em_min = float(scan.get("Em_min", 0.0))
    Em_max = float(scan.get("Em_max", 3.0))
    Em_num = int(scan.get("Em_num", 9))
    f_min  = float(scan.get("f_min", 0.0))
    f_max  = float(scan.get("f_max", 0.05))
    f_num  = int(scan.get("f_num", 9))

    Em_vals = np.linspace(Em_min, Em_max, Em_num)
    f_vals  = np.linspace(f_min, f_max, f_num)

    CV = np.full((len(f_vals), len(Em_vals)), np.nan, dtype=float)
    RR = np.full((len(f_vals), len(Em_vals)), np.nan, dtype=float)

    for iy, fv in enumerate(f_vals):
        for ix, Em in enumerate(Em_vals):
            cfg2 = deep_copy_cfg(cfg)
            cfg2["model"]["Em"] = float(Em)
            cfg2["model"]["f"]  = float(fv)

            Thist, Xhist = simulate_store_X(cfg2)
            met = compute_metrics(cfg2, Thist, Xhist)
            CV[iy, ix] = met["cv_mean"]
            RR[iy, ix] = met["kuramoto_R"]

            print(f"[map] f={fv:.5f} Em={Em:.3f}  CV={met['cv_mean']:.4f}  R={met['kuramoto_R']:.4f}")

    np.savez_compressed(os.path.join(out_raw, "phase_map_em_f__raw.npz"),
                        Em_vals=Em_vals, f_vals=f_vals, CV=CV, R=RR)

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__cv_mean.png"),
                 CV, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="Network CV mean (spike irregularity)")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__kuramoto_R.png"),
                 RR, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="Kuramoto R (phase coherence)")

    # -------------------------
    # 2) Frequency slice: R(f) for multiple Em
    # -------------------------
    sl = cfg.get("paper_outputs", {}).get("frequency_slice", {})
    f_slice_min = float(sl.get("f_min", 0.0))
    f_slice_max = float(sl.get("f_max", 0.05))
    f_slice_num = int(sl.get("f_num", 13))
    Em_list = sl.get("Em_list", [0.5, 1.5, 2.5])

    f_grid = np.linspace(f_slice_min, f_slice_max, f_slice_num)

    R_curves = []
    labels = []
    csv_path = os.path.join(out_slices, "slice_R_vs_f__multiple_Em.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["f"] + [f"R_Em_{Em}" for Em in Em_list])

        for Em in Em_list:
            Rvals = []
            for fv in f_grid:
                cfg2 = deep_copy_cfg(cfg)
                cfg2["model"]["Em"] = float(Em)
                cfg2["model"]["f"]  = float(fv)
                Thist, Xhist = simulate_store_X(cfg2)
                met = compute_metrics(cfg2, Thist, Xhist)
                Rvals.append(met["kuramoto_R"])
                print(f"[slice] Em={Em:.3f} f={fv:.5f} R={met['kuramoto_R']:.4f}")
            Rvals = np.array(Rvals, dtype=float)
            R_curves.append(Rvals)
            labels.append(f"Em={Em}")

        # write rows
        for i in range(len(f_grid)):
            row = [float(f_grid[i])] + [float(R_curves[j][i]) for j in range(len(Em_list))]
            w.writerow(row)

    save_lineplot(os.path.join(out_slices, "slice_R_vs_f__multiple_Em.png"),
                  f_grid, R_curves, labels,
                  xlabel="f", ylabel="Kuramoto R",
                  title="Frequency dependence of coherence (multiple Em)")

    # -------------------------
    # 3) Temperature sweep: R(T) and CV(T) at fixed (Em,f)
    # -------------------------
    ts = cfg.get("paper_outputs", {}).get("temperature_sweep", {})
    T_list = ts.get("T_list", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    Em_fix = float(ts.get("Em_fixed", 2.5))
    f_fix  = float(ts.get("f_fixed", 0.02))

    T_vals = []
    R_vals = []
    CV_vals = []

    csv_path_T = os.path.join(out_temp, "temperature_sweep__R_and_CV_vs_T.csv")
    with open(csv_path_T, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["T", "kuramoto_R", "cv_mean", "Em_fixed", "f_fixed"])

        for T in T_list:
            cfg2 = deep_copy_cfg(cfg)
            cfg2["model"]["T"]  = float(T)
            cfg2["model"]["Em"] = float(Em_fix)
            cfg2["model"]["f"]  = float(f_fix)
            Thist, Xhist = simulate_store_X(cfg2)
            met = compute_metrics(cfg2, Thist, Xhist)

            T_vals.append(float(T))
            R_vals.append(float(met["kuramoto_R"]))
            CV_vals.append(float(met["cv_mean"]))

            w.writerow([float(T), float(met["kuramoto_R"]), float(met["cv_mean"]), Em_fix, f_fix])
            print(f"[temp] T={T:.3f} Em={Em_fix:.3f} f={f_fix:.5f}  R={met['kuramoto_R']:.4f}  CV={met['cv_mean']:.4f}")

    save_twoaxis(os.path.join(out_temp, "temperature_sweep__R_and_CV_vs_T.png"),
                 np.array(T_vals), np.array(R_vals), np.array(CV_vals),
                 xlabel="T",
                 y1label="Kuramoto R",
                 y2label="CV mean",
                 title=f"Thermal modulation at Em={Em_fix}, f={f_fix}")

    # -------------------------
    # 4) Example outputs: time series + raster
    # -------------------------
    ex = cfg.get("paper_outputs", {}).get("examples", {})
    Em_ex = float(ex.get("Em_example", 2.5))
    f_ex  = float(ex.get("f_example", 0.02))
    T_ex  = float(ex.get("T_example", cfg["model"]["T"]))
    trace_i, trace_j = ex.get("trace_node", cfg["analysis"]["trace_node"])
    trace_i = int(trace_i); trace_j = int(trace_j)

    cfgE = deep_copy_cfg(cfg)
    cfgE["model"]["Em"] = Em_ex
    cfgE["model"]["f"]  = f_ex
    cfgE["model"]["T"]  = T_ex
    Thist, Xhist = simulate_store_X(cfgE)

    np.savez_compressed(os.path.join(out_raw, "example_run__raw.npz"),
                        t=Thist, X=Xhist,
                        Em=Em_ex, f=f_ex, T=T_ex)

    save_timeseries_plot(os.path.join(out_examples, "example_timeseries__center_node.png"),
                         Thist, Xhist[:, trace_i, trace_j],
                         title=f"Example x(t) at node ({trace_i},{trace_j})  Em={Em_ex}, f={f_ex}, T={T_ex}")

    burn_in = float(cfg["sim"]["burn_in"])
    thr = float(cfg["analysis"]["spike_threshold"])
    min_sep = float(cfg["analysis"]["min_spike_separation"])

    raster_from_spikes(os.path.join(out_examples, "example_raster__spike_events.png"),
                       Thist, Xhist, thr=thr, min_sep=min_sep, burn_in=burn_in, max_neurons=400)

    # -------------------------
    # Summary JSON
    # -------------------------
    summary = {
        "run_dir": run_dir,
        "outputs": {
            "maps_em_f": out_maps,
            "frequency_slices": out_slices,
            "temperature_sweep": out_temp,
            "examples": out_examples,
            "raw": out_raw
        },
        "phase_map": {
            "Em_range": [Em_min, Em_max, Em_num],
            "f_range": [f_min, f_max, f_num],
            "cv_min": float(np.nanmin(CV)),
            "cv_max": float(np.nanmax(CV)),
            "R_min": float(np.nanmin(RR)),
            "R_max": float(np.nanmax(RR))
        },
        "frequency_slice": {
            "Em_list": Em_list,
            "f_range": [f_slice_min, f_slice_max, f_slice_num]
        },
        "temperature_sweep": {
            "T_list": T_list,
            "Em_fixed": Em_fix,
            "f_fixed": f_fix
        },
        "example": {
            "Em_example": Em_ex,
            "f_example": f_ex,
            "T_example": T_ex,
            "trace_node": [trace_i, trace_j]
        }
    }
    write_json(os.path.join(run_dir, "run_summary.json"), summary)

    print("RUN_DIR:", run_dir)
    print("WROTE:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
