import numpy as np

def detect_spikes(t, x, thr=0.0, min_sep=0.5):
    """
    Detect upward threshold crossings with a refractory separation min_sep (in time units).
    Returns spike times (float list).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size < 2:
        return []

    # upward crossings: x[k-1] < thr and x[k] >= thr
    crossings = np.where((x[:-1] < thr) & (x[1:] >= thr))[0] + 1
    if crossings.size == 0:
        return []

    spikes = []
    last_t = -np.inf
    for idx in crossings:
        ts = float(t[idx])
        if ts - last_t >= float(min_sep):
            spikes.append(ts)
            last_t = ts
    return spikes

def _cv_from_spike_times(spike_times):
    """
    CV of ISIs. If fewer than 3 spikes -> NaN (needs >=2 ISIs for stable estimate).
    """
    if spike_times is None or len(spike_times) < 3:
        return np.nan
    isi = np.diff(np.array(spike_times, dtype=float))
    m = float(np.mean(isi))
    if m <= 0:
        return np.nan
    return float(np.std(isi) / m)

def activity_metrics(Thist, Xhist, thr=0.0, min_sep=0.5, burn_in=0.0, min_spikes_per_neuron=5):
    """
    Robust, publication-defendable metrics:
      - active_fraction: fraction of neurons with >= min_spikes_per_neuron spikes after burn-in
      - cv_active_mean / cv_active_median: CV computed only over active neurons
      - firing_rate_all: mean spikes/time over all neurons (after burn-in)
      - firing_rate_active: mean spikes/time over active neurons only
      - spike_count_mean / spike_count_median
    """
    t = np.asarray(Thist, dtype=float)
    X = np.asarray(Xhist, dtype=float)
    assert X.ndim == 3, "Xhist must be (T, nx, ny)"
    nx, ny = X.shape[1], X.shape[2]
    N = nx * ny

    mask = t >= float(burn_in)
    tt = t[mask]
    XX = X[mask]
    if tt.size < 2:
        return {
            "active_fraction": 0.0,
            "cv_active_mean": np.nan,
            "cv_active_median": np.nan,
            "firing_rate_all": 0.0,
            "firing_rate_active": np.nan,
            "spike_count_mean": 0.0,
            "spike_count_median": 0.0,
            "min_spikes_per_neuron": int(min_spikes_per_neuron),
            "n_neurons": int(N)
        }

    duration = float(tt[-1] - tt[0])
    if duration <= 0:
        duration = float(tt[-1] - tt[0] + 1e-12)

    spike_counts = np.zeros(N, dtype=int)
    cvs = np.full(N, np.nan, dtype=float)

    k = 0
    for i in range(nx):
        for j in range(ny):
            st = detect_spikes(tt, XX[:, i, j], thr=thr, min_sep=min_sep)
            spike_counts[k] = len(st)
            cvs[k] = _cv_from_spike_times(st)
            k += 1

    min_sp = int(min_spikes_per_neuron)
    active = spike_counts >= min_sp
    active_fraction = float(np.mean(active)) if N > 0 else 0.0

    # firing rates
    firing_rate_all = float(np.mean(spike_counts) / duration)
    if np.any(active):
        firing_rate_active = float(np.mean(spike_counts[active]) / duration)
        cv_active_mean = float(np.nanmean(cvs[active]))
        cv_active_median = float(np.nanmedian(cvs[active]))
    else:
        firing_rate_active = np.nan
        cv_active_mean = np.nan
        cv_active_median = np.nan

    return {
        "active_fraction": active_fraction,
        "cv_active_mean": cv_active_mean,
        "cv_active_median": cv_active_median,
        "firing_rate_all": firing_rate_all,
        "firing_rate_active": firing_rate_active,
        "spike_count_mean": float(np.mean(spike_counts)),
        "spike_count_median": float(np.median(spike_counts)),
        "min_spikes_per_neuron": min_sp,
        "n_neurons": int(N)
    }
