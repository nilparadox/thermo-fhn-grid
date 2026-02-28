import os
import numpy as np
import matplotlib.pyplot as plt

def save_timeseries_plot(out_png: str, t: np.ndarray, x: np.ndarray, title: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("x (membrane potential)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_heatmap(out_png: str, Z: np.ndarray, xvals: np.ndarray, yvals: np.ndarray,
                 xlabel: str, ylabel: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure()
    plt.imshow(Z, origin="lower", aspect="auto",
               extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_bifurcation(out_png: str, pvals: np.ndarray, maxima: list, xlabel: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure()
    for pv, xs in zip(pvals, maxima):
        if len(xs) == 0:
            continue
        plt.plot([pv]*len(xs), xs, ".", markersize=1)
    plt.xlabel(xlabel)
    plt.ylabel("x maxima")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
