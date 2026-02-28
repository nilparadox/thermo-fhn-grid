import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .grid import laplacian_2d

@dataclass
class Params:
    a: float
    c: float
    xi: float
    b: float
    T: float
    I: float
    A: float
    omega: float
    r: float
    k: float
    Em: float
    f: float

@dataclass
class Noise:
    enable: bool
    sigma_x: float
    sigma_y: float
    sigma_E: float
    type: str

def e_ext(t: float, Em: float, f: float) -> float:
    return Em * np.sin(2.0 * np.pi * f * t)

def drift(x: np.ndarray, y: np.ndarray, E: np.ndarray, t: float, p: Params,
          Kx: float, boundary: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Paper model (thermosensitive FHN + electric field variable) with network coupling on x.
    # dx/dt = x(1-xi) - x^3/3 - y + I + A cos(omega t) + Kx * Laplacian(x)
    # dy/dt = c [ x + a - b exp(1/T) y ] + r E
    # dE/dt = k y + Eext(t)
    stim = p.I + p.A * np.cos(p.omega * t)
    Lx = laplacian_2d(x, boundary=boundary)

    dx = x * (1.0 - p.xi) - (x**3) / 3.0 - y + stim + Kx * Lx
    dy = p.c * (x + p.a - p.b * np.exp(1.0 / p.T) * y) + p.r * E
    dE = p.k * y + e_ext(t, p.Em, p.f)
    return dx, dy, dE

def step_em(x: np.ndarray, y: np.ndarray, E: np.ndarray, t: float, dt: float,
            p: Params, Kx: float, boundary: str,
            noise: Noise, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dE = drift(x, y, E, t, p, Kx, boundary)

    if noise.enable:
        if noise.type != "additive_gaussian":
            raise ValueError(f"Unsupported noise.type={noise.type}")
        # Euler–Maruyama: X_{n+1}=X_n+f dt + sigma sqrt(dt) N(0,1)
        x = x + dx * dt + noise.sigma_x * np.sqrt(dt) * rng.standard_normal(size=x.shape)
        y = y + dy * dt + noise.sigma_y * np.sqrt(dt) * rng.standard_normal(size=y.shape)
        E = E + dE * dt + noise.sigma_E * np.sqrt(dt) * rng.standard_normal(size=E.shape)
    else:
        x = x + dx * dt
        y = y + dy * dt
        E = E + dE * dt

    return x, y, E

def build_params(cfg: Dict[str, Any]) -> Params:
    m = cfg["model"]
    return Params(
        a=float(m["a"]), c=float(m["c"]), xi=float(m["xi"]),
        b=float(m["b"]), T=float(m["T"]), I=float(m["I"]),
        A=float(m["A"]), omega=float(m["omega"]),
        r=float(m["r"]), k=float(m["k"]),
        Em=float(m["Em"]), f=float(m["f"])
    )

def build_noise(cfg: Dict[str, Any]) -> Noise:
    n = cfg["noise"]
    return Noise(
        enable=bool(n["enable"]),
        sigma_x=float(n["sigma_x"]),
        sigma_y=float(n["sigma_y"]),
        sigma_E=float(n["sigma_E"]),
        type=str(n["type"])
    )
