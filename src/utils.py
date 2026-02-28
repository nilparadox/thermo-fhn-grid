import os, json, hashlib, datetime
from typing import Any, Dict, Tuple

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:10]

def make_run_dir(base: str, cfg: Dict[str, Any]) -> str:
    tag = now_tag()
    h = stable_hash(cfg)
    m = cfg.get("model", {})
    sim = cfg.get("sim", {})
    nx = cfg.get("grid", {}).get("nx", "?")
    ny = cfg.get("grid", {}).get("ny", "?")
    Em = m.get("Em", "?")
    f = m.get("f", "?")
    T = m.get("T", "?")
    omega = m.get("omega", "?")
    dt = sim.get("dt", "?")
    Kx = cfg.get("coupling", {}).get("Kx", "?")

    name = f"{tag}__grid{nx}x{ny}__T{T}__Em{Em}__f{f}__w{omega}__Kx{Kx}__dt{dt}__{h}"
    run_dir = os.path.join(base, name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def split_node_index(i: int, nx: int) -> Tuple[int, int]:
    return (i // nx, i % nx)
