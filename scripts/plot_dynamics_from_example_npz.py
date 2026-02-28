import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

from src.robust_metrics import detect_spikes

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_latest_run():
    runs = sorted(glob.glob("runs/*"), key=os.path.getmtime)
    if not runs:
        raise RuntimeError("No runs/ found.")
    return runs[-1]

def load_example_npz(run_dir):
    npz_path = os.path.join(run_dir, "raw", "example_run__seed0_raw.npz")
    if not os.path.isfile(npz_path):
        raise RuntimeError(f"Missing: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    t = d["t"].astype(float)
    X = d["X"].astype(float)  # shape (T, nx, ny)
    meta = {}
    for k in ["seed", "Em", "f", "T"]:
        if k in d.files:
            try:
                meta[k] = float(d[k])
            except Exception:
                meta[k] = d[k].item()
    return npz_path, t, X, meta

def spikes_for_all(tt, XX, thr=0.0, min_sep=0.5):
    nx, ny = XX.shape[1], XX.shape[2]
    spikes = [[None for _ in range(ny)] for __ in range(nx)]
    for i in range(nx):
        for j in range(ny):
            spikes[i][j] = np.array(detect_spikes(tt, XX[:, i, j], thr=thr, min_sep=min_sep), dtype=float)
    return spikes

def phase_from_spikes_at_times(tt, spike_times):
    """
    Given spike_times array for one neuron, compute phase phi(t) on tt.
    Between spikes k and k+1: phi = 2π*(t-tk)/(tk+1-tk).
    Outside spike range -> NaN.
    """
    st = np.asarray(spike_times, dtype=float)
    phi = np.full(tt.shape, np.nan, dtype=float)
    if st.size < 2:
        return phi

    # for each t, find index of the next spike
    idx_next = np.searchsorted(st, tt, side="right")
    # next spike is st[idx_next] (if idx_next < len)
    # prev spike is st[idx_next - 1] (if idx_next > 0)
    valid = (idx_next > 0) & (idx_next < st.size)
    if not np.any(valid):
        return phi

    prev = st[idx_next[valid] - 1]
    nex  = st[idx_next[valid]]
    denom = (nex - prev)
    denom[denom <= 0] = np.nan
    frac = (tt[valid] - prev) / denom
    phi[valid] = 2.0 * np.pi * frac
    return phi

def kuramoto_R_t(tt, spikes_grid, max_neurons=None):
    """
    Compute Kuramoto order parameter R(t) using spike-defined phase for each neuron.
    If max_neurons is set, subsample neurons uniformly for speed.
    """
    nx = len(spikes_grid)
    ny = len(spikes_grid[0])
    N = nx * ny

    if max_neurons is None or max_neurons >= N:
        indices = np.arange(N)
    else:
        indices = np.linspace(0, N-1, int(max_neurons)).astype(int)

    # build exp(i phi) sum without storing huge (T,N) matrix
    sum_re = np.zeros(tt.shape, dtype=float)
    sum_im = np.zeros(tt.shape, dtype=float)
    count  = np.zeros(tt.shape, dtype=float)

    for idx in indices:
        i = idx // ny
        j = idx % ny
        phi = phase_from_spikes_at_times(tt, spikes_grid[i][j])
        m = ~np.isnan(phi)
        if np.any(m):
            sum_re[m] += np.cos(phi[m])
            sum_im[m] += np.sin(phi[m])
            count[m]  += 1.0

    R = np.full(tt.shape, np.nan, dtype=float)
    m = count > 0
    R[m] = np.sqrt(sum_re[m]**2 + sum_im[m]**2) / count[m]
    return R

def firing_rate_map(spikes_grid, t0, t1):
    nx = len(spikes_grid)
    ny = len(spikes_grid[0])
    dur = float(t1 - t0)
    fr = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            st = spikes_grid[i][j]
            if st is None or len(st) == 0:
                fr[i, j] = 0.0
            else:
                fr[i, j] = np.sum((st >= t0) & (st <= t1)) / dur
    return fr

def phase_snapshot_map(tt, spikes_grid, t_snapshot):
    nx = len(spikes_grid)
    ny = len(spikes_grid[0])
    phi_map = np.full((nx, ny), np.nan, dtype=float)
    for i in range(nx):
        for j in range(ny):
            st = spikes_grid[i][j]
            if st is None or len(st) < 2:
                continue
            # find surrounding spikes
            k = np.searchsorted(st, t_snapshot, side="right")
            if k <= 0 or k >= len(st):
                continue
            prev = st[k-1]; nex = st[k]
            if nex <= prev:
                continue
            frac = (t_snapshot - prev) / (nex - prev)
            phi_map[i, j] = 2.0 * np.pi * frac
    return phi_map

def save_heatmap(path, Z, title, xlabel="j", ylabel="i", vmin=None, vmax=None):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(6,5))
    im = plt.imshow(Z, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def save_line(path, x, y, title, xlabel, ylabel):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(7,4))
    plt.plot(x, y, linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def save_two_lines(path, x, y1, y2, title, xlabel, y1label, y2label):
    ensure_dir(os.path.dirname(path))
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(x, y1, linewidth=1.2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.set_title(title)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, linewidth=1.2)
    ax2.set_ylabel(y2label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="runs/<...> (default: latest)")
    ap.add_argument("--thr", type=float, default=0.0, help="spike threshold for x")
    ap.add_argument("--min_sep", type=float, default=0.5, help="min separation between spikes (time units)")
    ap.add_argument("--burn_in", type=float, default=200.0, help="burn-in time used for steady-window analysis")
    ap.add_argument("--R_subsample", type=int, default=625, help="neurons used for R(t) (<=625 for full grid)")
    ap.add_argument("--node_a", type=str, default="3,3", help="node a as i,j")
    ap.add_argument("--node_b", type=str, default="21,21", help="node b as i,j")
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run()
    npz_path, t, X, meta = load_example_npz(run_dir)

    # output folder
    out_dir = os.path.join(run_dir, "examples", "derived_dynamics_from_raw")
    ensure_dir(out_dir)

    # parse nodes
    ia, ja = [int(z) for z in args.node_a.split(",")]
    ib, jb = [int(z) for z in args.node_b.split(",")]

    # spikes
    print("Loading:", npz_path)
    print("t shape:", t.shape, "X shape:", X.shape, "meta:", meta)
    print("Computing spikes for all neurons...")
    spikes = spikes_for_all(t, X, thr=args.thr, min_sep=args.min_sep)

    # mean-field
    xbar = X.mean(axis=(1,2))
    save_line(
        os.path.join(out_dir, "dynamics_mean_field_xbar_vs_time.png"),
        t, xbar,
        title="Mean-field activity: x̄(t)",
        xlabel="t",
        ylabel="x̄(t)"
    )

    # R(t)
    print("Computing R(t)... (subsample =", args.R_subsample, ")")
    Rt = kuramoto_R_t(t, spikes, max_neurons=args.R_subsample)
    save_line(
        os.path.join(out_dir, "dynamics_kuramoto_R_vs_time.png"),
        t, Rt,
        title="Kuramoto coherence over time: R(t)",
        xlabel="t",
        ylabel="R(t)"
    )

    # single-node traces for two nodes
    save_line(
        os.path.join(out_dir, f"dynamics_node_trace_x_vs_time__node_{ia}_{ja}.png"),
        t, X[:, ia, ja],
        title=f"Single-node trace x(t) at node ({ia},{ja})",
        xlabel="t",
        ylabel="x(t)"
    )
    save_line(
        os.path.join(out_dir, f"dynamics_node_trace_x_vs_time__node_{ib}_{jb}.png"),
        t, X[:, ib, jb],
        title=f"Single-node trace x(t) at node ({ib},{jb})",
        xlabel="t",
        ylabel="x(t)"
    )

    # phase difference Δphi(t) between two nodes
    print("Computing Δphi(t) between nodes...")
    phi_a = phase_from_spikes_at_times(t, spikes[ia][ja])
    phi_b = phase_from_spikes_at_times(t, spikes[ib][jb])
    dphi = np.full(t.shape, np.nan, dtype=float)
    m = (~np.isnan(phi_a)) & (~np.isnan(phi_b))
    if np.any(m):
        raw = phi_a[m] - phi_b[m]
        # wrap to [-pi, pi]
        dphi[m] = (raw + np.pi) % (2*np.pi) - np.pi

    save_line(
        os.path.join(out_dir, f"dynamics_phase_difference_dphi_vs_time__node_{ia}_{ja}__node_{ib}_{jb}.png"),
        t, dphi,
        title=f"Phase difference Δφ(t) between nodes ({ia},{ja}) and ({ib},{jb})",
        xlabel="t",
        ylabel="Δφ(t) (wrapped)"
    )

    # firing-rate map in steady window
    t0 = float(args.burn_in)
    t1 = float(t[-1])
    fr_map = firing_rate_map(spikes, t0, t1)
    save_heatmap(
        os.path.join(out_dir, "spatial_firing_rate_map__steady_window.png"),
        fr_map,
        title=f"Firing rate map (steady window): t∈[{t0:.1f},{t1:.1f}]",
        xlabel="j",
        ylabel="i"
    )

    # phase snapshot map near end
    t_snap = float(0.9*t[-1])
    phi_map = phase_snapshot_map(t, spikes, t_snap)
    save_heatmap(
        os.path.join(out_dir, "spatial_phase_snapshot_map__late_time.png"),
        phi_map,
        title=f"Phase snapshot map at t={t_snap:.2f}",
        xlabel="j",
        ylabel="i",
        vmin=0.0,
        vmax=2*np.pi
    )

    # write a small metadata JSON so you can cite exact settings
    meta_out = {
        "run_dir": run_dir,
        "npz_path": npz_path,
        "meta_from_npz": meta,
        "spike_threshold": args.thr,
        "min_spike_separation": args.min_sep,
        "burn_in_for_steady_window": args.burn_in,
        "R_subsample_neurons": args.R_subsample,
        "phase_nodes": {"a":[ia,ja], "b":[ib,jb]},
        "steady_window": [t0, t1],
        "phase_snapshot_time": t_snap,
        "outputs_dir": out_dir
    }
    with open(os.path.join(out_dir, "derived_dynamics_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print("WROTE dynamics plots to:", out_dir)

if __name__ == "__main__":
    main()
