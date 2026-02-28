import os, json, csv
import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_json, write_json, make_run_dir
from src.model import build_params, build_noise, step_em
from src.analysis import network_cv, kuramoto_R  # keep your current metrics
from src.robust_metrics import activity_metrics  # new defendable metrics
from src.plotting import save_heatmap, save_timeseries_plot

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def deep_copy_cfg(cfg):
    return json.loads(json.dumps(cfg))

def simulate_store_X(cfg):
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

def compute_all_metrics(cfg, Thist, Xhist, min_spikes_per_neuron=5):
    burn_in = float(cfg["sim"]["burn_in"])
    thr = float(cfg["analysis"]["spike_threshold"])
    min_sep = float(cfg["analysis"]["min_spike_separation"])

    # existing metrics (keep for continuity)
    cv_stats = network_cv(Thist, Xhist, thr=thr, min_sep=min_sep, burn_in=burn_in)
    R = kuramoto_R(Thist, Xhist, thr=thr, burn_in=burn_in)

    # robust metrics
    robust = activity_metrics(
        Thist, Xhist,
        thr=thr, min_sep=min_sep, burn_in=burn_in,
        min_spikes_per_neuron=min_spikes_per_neuron
    )

    out = {
        "kuramoto_R": float(R),
        "cv_mean": float(cv_stats["cv_mean"]),
        "cv_median": float(cv_stats["cv_median"]),
        "cv_n": int(cv_stats["cv_n"]),
    }
    out.update(robust)
    return out

def save_line_with_error(out_png, x, y_mean, y_std, labels, xlabel, ylabel, title):
    ensure_dir(os.path.dirname(out_png))
    plt.figure()
    for ym, ys, lab in zip(y_mean, y_std, labels):
        plt.errorbar(x, ym, yerr=ys, marker="o", capsize=3, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_simple_line(out_png, x, y, xlabel, ylabel, title):
    ensure_dir(os.path.dirname(out_png))
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_two_lines_with_error(out_png, x, y1m, y1s, y2m, y2s, xlabel, y1label, y2label, title):
    ensure_dir(os.path.dirname(out_png))
    fig, ax1 = plt.subplots()
    ax1.errorbar(x, y1m, yerr=y1s, marker="o", capsize=3)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.errorbar(x, y2m, yerr=y2s, marker="s", capsize=3)
    ax2.set_ylabel(y2label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="grid_network_outputs_replicates")
    args = ap.parse_args()

    cfg = read_json(args.config)

    rep = cfg.get("replicates", {})
    seeds = rep.get("seeds", [12345, 12346, 12347, 12348, 12349])
    min_spikes = int(rep.get("min_spikes_per_neuron", 5))

    run_cfg = deep_copy_cfg(cfg)
    run_cfg["run_label"] = args.label
    run_dir = make_run_dir("runs", run_cfg)

    out_maps = os.path.join(run_dir, "maps_em_f")
    out_slices = os.path.join(run_dir, "slices_frequency")
    out_temp = os.path.join(run_dir, "temperature_sweep")
    out_examples = os.path.join(run_dir, "examples")
    out_raw = os.path.join(run_dir, "raw")
    for d in [out_maps, out_slices, out_temp, out_examples, out_raw]:
        ensure_dir(d)

    write_json(os.path.join(run_dir, "config_used.json"), cfg)
    write_json(os.path.join(run_dir, "replicate_seeds.json"), {"seeds": seeds, "min_spikes_per_neuron": min_spikes})

    # =========================
    # 1) Phase map (Em,f): mean/std of R, CV_active, active_fraction
    # =========================
    scan = cfg.get("paper_outputs", {}).get("em_f_map", {})
    Em_min = float(scan.get("Em_min", 0.0))
    Em_max = float(scan.get("Em_max", 3.0))
    Em_num = int(scan.get("Em_num", 9))
    f_min  = float(scan.get("f_min", 0.0))
    f_max  = float(scan.get("f_max", 0.05))
    f_num  = int(scan.get("f_num", 9))

    Em_vals = np.linspace(Em_min, Em_max, Em_num)
    f_vals  = np.linspace(f_min, f_max, f_num)

    # arrays: (S, Fy, Ex)
    S = len(seeds)
    R_all  = np.full((S, len(f_vals), len(Em_vals)), np.nan, dtype=float)
    CVa_all = np.full_like(R_all, np.nan)
    AF_all  = np.full_like(R_all, np.nan)

    for si, seed in enumerate(seeds):
        for iy, fv in enumerate(f_vals):
            for ix, Em in enumerate(Em_vals):
                cfg2 = deep_copy_cfg(cfg)
                cfg2["sim"]["seed"] = int(seed)
                cfg2["model"]["Em"] = float(Em)
                cfg2["model"]["f"]  = float(fv)

                Thist, Xhist = simulate_store_X(cfg2)
                met = compute_all_metrics(cfg2, Thist, Xhist, min_spikes_per_neuron=min_spikes)

                R_all[si, iy, ix] = met["kuramoto_R"]
                CVa_all[si, iy, ix] = met["cv_active_mean"]
                AF_all[si, iy, ix] = met["active_fraction"]

                print(f"[map][seed={seed}] f={fv:.5f} Em={Em:.3f}  R={met['kuramoto_R']:.4f}  CV_active={met['cv_active_mean']:.4f}  active={met['active_fraction']:.3f}")

    R_mean  = np.nanmean(R_all, axis=0)
    R_std   = np.nanstd(R_all, axis=0)
    CVa_mean = np.nanmean(CVa_all, axis=0)
    CVa_std  = np.nanstd(CVa_all, axis=0)
    AF_mean  = np.nanmean(AF_all, axis=0)
    AF_std   = np.nanstd(AF_all, axis=0)

    np.savez_compressed(os.path.join(out_raw, "phase_map_em_f__replicates_raw.npz"),
                        Em_vals=Em_vals, f_vals=f_vals,
                        seeds=np.array(seeds, dtype=int),
                        R_all=R_all, CV_active_all=CVa_all, active_fraction_all=AF_all,
                        R_mean=R_mean, R_std=R_std,
                        CV_active_mean=CVa_mean, CV_active_std=CVa_std,
                        active_fraction_mean=AF_mean, active_fraction_std=AF_std)

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__kuramoto_R__mean.png"),
                 R_mean, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="Kuramoto R (mean over seeds)")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__kuramoto_R__std.png"),
                 R_std, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="Kuramoto R (std over seeds)")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__cv_active_mean__mean.png"),
                 CVa_mean, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="CV mean over active neurons (mean over seeds)")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__cv_active_mean__std.png"),
                 CVa_std, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title="CV mean over active neurons (std over seeds)")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__active_fraction__mean.png"),
                 AF_mean, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title=f"Active fraction (>= {min_spikes} spikes) mean over seeds")

    save_heatmap(os.path.join(out_maps, "phase_map_em_f__active_fraction__std.png"),
                 AF_std, Em_vals, f_vals, xlabel="Em", ylabel="f",
                 title=f"Active fraction (>= {min_spikes} spikes) std over seeds")

    # =========================
    # 2) Frequency slice: R(f) mean±std for multiple Em + CSVs
    # =========================
    sl = cfg.get("paper_outputs", {}).get("frequency_slice", {})
    f_slice_min = float(sl.get("f_min", 0.0))
    f_slice_max = float(sl.get("f_max", 0.05))
    f_slice_num = int(sl.get("f_num", 13))
    Em_list = sl.get("Em_list", [0.5, 1.5, 2.5])
    f_grid = np.linspace(f_slice_min, f_slice_max, f_slice_num)

    # shape: (S, E, F)
    R_sf = np.full((S, len(Em_list), len(f_grid)), np.nan, dtype=float)
    AF_sf = np.full_like(R_sf, np.nan)
    CVa_sf = np.full_like(R_sf, np.nan)

    for si, seed in enumerate(seeds):
        for ei, Em in enumerate(Em_list):
            for fi, fv in enumerate(f_grid):
                cfg2 = deep_copy_cfg(cfg)
                cfg2["sim"]["seed"] = int(seed)
                cfg2["model"]["Em"] = float(Em)
                cfg2["model"]["f"]  = float(fv)

                Thist, Xhist = simulate_store_X(cfg2)
                met = compute_all_metrics(cfg2, Thist, Xhist, min_spikes_per_neuron=min_spikes)

                R_sf[si, ei, fi] = met["kuramoto_R"]
                AF_sf[si, ei, fi] = met["active_fraction"]
                CVa_sf[si, ei, fi] = met["cv_active_mean"]

                print(f"[slice][seed={seed}] Em={Em:.3f} f={fv:.5f}  R={met['kuramoto_R']:.4f} active={met['active_fraction']:.3f} CV_active={met['cv_active_mean']:.4f}")

    Rm = np.nanmean(R_sf, axis=0); Rs = np.nanstd(R_sf, axis=0)
    AFm = np.nanmean(AF_sf, axis=0); AFs = np.nanstd(AF_sf, axis=0)
    CVm = np.nanmean(CVa_sf, axis=0); CVs = np.nanstd(CVa_sf, axis=0)

    np.savez_compressed(os.path.join(out_raw, "slice_R_f__replicates_raw.npz"),
                        seeds=np.array(seeds, dtype=int), f_grid=f_grid, Em_list=np.array(Em_list, dtype=float),
                        R_all=R_sf, R_mean=Rm, R_std=Rs,
                        active_fraction_all=AF_sf, active_fraction_mean=AFm, active_fraction_std=AFs,
                        cv_active_all=CVa_sf, cv_active_mean=CVm, cv_active_std=CVs)

    # CSV: R mean/std
    csv_R = os.path.join(out_slices, "slice_R_vs_f__multiple_Em__mean_std.csv")
    with open(csv_R, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        header = ["f"]
        for Em in Em_list:
            header += [f"R_mean_Em_{Em}", f"R_std_Em_{Em}"]
        w.writerow(header)
        for fi in range(len(f_grid)):
            row = [float(f_grid[fi])]
            for ei in range(len(Em_list)):
                row += [float(Rm[ei, fi]), float(Rs[ei, fi])]
            w.writerow(row)

    # CSV: active fraction mean/std
    csv_AF = os.path.join(out_slices, "slice_active_fraction_vs_f__multiple_Em__mean_std.csv")
    with open(csv_AF, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        header = ["f"]
        for Em in Em_list:
            header += [f"active_fraction_mean_Em_{Em}", f"active_fraction_std_Em_{Em}"]
        w.writerow(header)
        for fi in range(len(f_grid)):
            row = [float(f_grid[fi])]
            for ei in range(len(Em_list)):
                row += [float(AFm[ei, fi]), float(AFs[ei, fi])]
            w.writerow(row)

    labels = [f"Em={Em}" for Em in Em_list]
    save_line_with_error(os.path.join(out_slices, "slice_R_vs_f__multiple_Em__mean_std.png"),
                         f_grid, [Rm[i] for i in range(len(Em_list))], [Rs[i] for i in range(len(Em_list))],
                         labels, xlabel="f", ylabel="Kuramoto R",
                         title="Frequency dependence of coherence (mean ± std over seeds)")

    save_line_with_error(os.path.join(out_slices, "slice_active_fraction_vs_f__multiple_Em__mean_std.png"),
                         f_grid, [AFm[i] for i in range(len(Em_list))], [AFs[i] for i in range(len(Em_list))],
                         labels, xlabel="f", ylabel="Active fraction",
                         title=f"Active fraction (>= {min_spikes} spikes) vs frequency (mean ± std over seeds)")

    # =========================
    # 3) Temperature sweep: R(T), CV_active(T), active_fraction(T) mean±std
    # =========================
    ts = cfg.get("paper_outputs", {}).get("temperature_sweep", {})
    T_list = ts.get("T_list", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    Em_fix = float(ts.get("Em_fixed", 2.5))
    f_fix  = float(ts.get("f_fixed", 0.02))
    T_vals = np.array([float(T) for T in T_list], dtype=float)

    R_T = np.full((S, len(T_vals)), np.nan, dtype=float)
    CV_T = np.full_like(R_T, np.nan)
    AF_T = np.full_like(R_T, np.nan)

    for si, seed in enumerate(seeds):
        for ti, T in enumerate(T_vals):
            cfg2 = deep_copy_cfg(cfg)
            cfg2["sim"]["seed"] = int(seed)
            cfg2["model"]["T"] = float(T)
            cfg2["model"]["Em"] = float(Em_fix)
            cfg2["model"]["f"]  = float(f_fix)

            Thist, Xhist = simulate_store_X(cfg2)
            met = compute_all_metrics(cfg2, Thist, Xhist, min_spikes_per_neuron=min_spikes)

            R_T[si, ti] = met["kuramoto_R"]
            CV_T[si, ti] = met["cv_active_mean"]
            AF_T[si, ti] = met["active_fraction"]

            print(f"[temp][seed={seed}] T={T:.3f} Em={Em_fix:.3f} f={f_fix:.5f}  R={met['kuramoto_R']:.4f}  CV_active={met['cv_active_mean']:.4f} active={met['active_fraction']:.3f}")

    RmT = np.nanmean(R_T, axis=0); RsT = np.nanstd(R_T, axis=0)
    CVmT = np.nanmean(CV_T, axis=0); CVsT = np.nanstd(CV_T, axis=0)
    AFmT = np.nanmean(AF_T, axis=0); AFsT = np.nanstd(AF_T, axis=0)

    np.savez_compressed(os.path.join(out_raw, "temperature_sweep__replicates_raw.npz"),
                        seeds=np.array(seeds, dtype=int),
                        T_vals=T_vals, Em_fixed=Em_fix, f_fixed=f_fix,
                        R_all=R_T, R_mean=RmT, R_std=RsT,
                        cv_active_all=CV_T, cv_active_mean=CVmT, cv_active_std=CVsT,
                        active_fraction_all=AF_T, active_fraction_mean=AFmT, active_fraction_std=AFsT)

    csv_T = os.path.join(out_temp, "temperature_sweep__mean_std.csv")
    with open(csv_T, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["T", "R_mean", "R_std", "cv_active_mean", "cv_active_std", "active_fraction_mean", "active_fraction_std", "Em_fixed", "f_fixed", "min_spikes_per_neuron"])
        for i in range(len(T_vals)):
            w.writerow([float(T_vals[i]), float(RmT[i]), float(RsT[i]),
                        float(CVmT[i]), float(CVsT[i]),
                        float(AFmT[i]), float(AFsT[i]),
                        float(Em_fix), float(f_fix), int(min_spikes)])

    save_two_lines_with_error(os.path.join(out_temp, "temperature_sweep__R_and_CV_active__mean_std.png"),
                              T_vals, RmT, RsT, CVmT, CVsT,
                              xlabel="T", y1label="Kuramoto R", y2label="CV (active neurons)",
                              title=f"Thermal modulation at Em={Em_fix}, f={f_fix} (mean ± std over seeds)")

    save_line_with_error(os.path.join(out_temp, "temperature_sweep__active_fraction__mean_std.png"),
                         T_vals, [AFmT], [AFsT], [f"active fraction (>= {min_spikes} spikes)"],
                         xlabel="T", ylabel="Active fraction",
                         title=f"Thermal modulation of activity (mean ± std over seeds)")

    # =========================
    # 4) Examples (use the first seed as representative)
    # =========================
    ex = cfg.get("paper_outputs", {}).get("examples", {})
    Em_ex = float(ex.get("Em_example", 2.5))
    f_ex  = float(ex.get("f_example", 0.02))
    T_ex  = float(ex.get("T_example", cfg["model"]["T"]))
    trace_i, trace_j = ex.get("trace_node", cfg["analysis"]["trace_node"])
    trace_i = int(trace_i); trace_j = int(trace_j)

    cfgE = deep_copy_cfg(cfg)
    cfgE["sim"]["seed"] = int(seeds[0])
    cfgE["model"]["Em"] = Em_ex
    cfgE["model"]["f"]  = f_ex
    cfgE["model"]["T"]  = T_ex

    Thist, Xhist = simulate_store_X(cfgE)
    np.savez_compressed(os.path.join(out_raw, "example_run__seed0_raw.npz"),
                        t=Thist, X=Xhist, seed=int(seeds[0]), Em=Em_ex, f=f_ex, T=T_ex)

    save_timeseries_plot(os.path.join(out_examples, "example_timeseries__center_node.png"),
                         Thist, Xhist[:, trace_i, trace_j],
                         title=f"Example x(t) at node ({trace_i},{trace_j})  Em={Em_ex}, f={f_ex}, T={T_ex}, seed={seeds[0]}")

    # simple raster (subsample neurons for readability)
    burn_in = float(cfg["sim"]["burn_in"])
    thr = float(cfg["analysis"]["spike_threshold"])
    min_sep = float(cfg["analysis"]["min_spike_separation"])
    mask = Thist >= burn_in
    tt = Thist[mask]
    XX = Xhist[mask]
    nx, ny = XX.shape[1], XX.shape[2]
    N = nx * ny
    max_neurons = int(ex.get("raster_max_neurons", 400))

    if N <= max_neurons:
        indices = np.arange(N)
    else:
        indices = np.linspace(0, N-1, max_neurons).astype(int)

    spike_t = []
    spike_y = []
    from src.robust_metrics import detect_spikes as _ds
    for row, idx in enumerate(indices):
        i = idx // ny
        j = idx % ny
        st = _ds(tt, XX[:, i, j], thr=thr, min_sep=min_sep)
        for s in st:
            spike_t.append(s)
            spike_y.append(row)

    plt.figure(figsize=(8,4))
    plt.plot(spike_t, spike_y, "k.", markersize=1.5)
    plt.xlabel("t")
    plt.ylabel("neuron (subsample index)")
    plt.title("Spike raster (subsampled neurons)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_examples, "example_raster__spike_events.png"), dpi=200)
    plt.close()

    # =========================
    # Summary
    # =========================
    summary = {
        "run_dir": run_dir,
        "seeds": seeds,
        "min_spikes_per_neuron": min_spikes,
        "outputs": {
            "maps_em_f": out_maps,
            "frequency_slices": out_slices,
            "temperature_sweep": out_temp,
            "examples": out_examples,
            "raw": out_raw
        },
        "phase_map_ranges": {
            "R_mean_min": float(np.nanmin(R_mean)),
            "R_mean_max": float(np.nanmax(R_mean)),
            "CV_active_mean_min": float(np.nanmin(CVa_mean)),
            "CV_active_mean_max": float(np.nanmax(CVa_mean)),
            "active_fraction_mean_min": float(np.nanmin(AF_mean)),
            "active_fraction_mean_max": float(np.nanmax(AF_mean))
        }
    }
    write_json(os.path.join(run_dir, "run_summary.json"), summary)

    print("RUN_DIR:", run_dir)
    print("WROTE:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
