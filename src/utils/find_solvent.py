from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass(frozen=True)
class SolventPeak:
    name: str
    h_peaks: List[Tuple[float, str]]
    c_peaks: List[float]


def _norm_mult(m: str) -> str:
    return (m or "").strip().lower()


def _assign_matches_1d(targets: List[float], pool: List[float], tol: float):
    used = [False] * len(pool)
    mapping: List[Optional[int]] = [None] * len(targets)

    for i, t in enumerate(targets):
        best_j = None
        best_delta = None
        for j, p in enumerate(pool):
            if used[j]:
                continue
            d = abs(t - p)
            if d <= tol and (best_delta is None or d < best_delta):
                best_delta = d
                best_j = j
        if best_j is not None:
            used[best_j] = True
            mapping[i] = best_j

    matched = sum(1 for x in mapping if x is not None)
    return matched, mapping


def _assign_matches_h(solvent_h: List[Tuple[float, str]], user_h: List[Tuple[float, str]], tol_h: float):
    used = [False] * len(user_h)
    mapping: List[Optional[int]] = [None] * len(solvent_h)

    for i, (s_shift, s_mult) in enumerate(solvent_h):
        sm = _norm_mult(s_mult)
        best_j = None
        best_delta = None

        for j, (u_shift, u_mult) in enumerate(user_h):
            if used[j]:
                continue
            if _norm_mult(u_mult) != sm:
                continue
            d = abs(s_shift - u_shift)
            if d <= tol_h and (best_delta is None or d < best_delta):
                best_delta = d
                best_j = j

        if best_j is not None:
            used[best_j] = True
            mapping[i] = best_j

    matched = sum(1 for x in mapping if x is not None)
    return matched, mapping


# Data source: Babij et al., Org. Process Res. Dev. 2016, 20, 661–667
SOLVENT_LIBRARY_BY_NMR_SOLVENT = {
    "cdcl3": [
        {"name": "CDCl3 residual", "h": [(7.26, "s")], "c": [77.06]},
        {"name": "CDCl3-1", "h": [(7.26, "s")], "c": [77.37]},
        {"name": "CDCl3-2", "h": [(7.26, "s")], "c": [76.73]},
        {"name": "water", "h": [(1.56, "s")], "c": []}, # water can only be detected by H-peak
        {"name": "acetone", "h": [(2.17, "s")], "c": [207.07, 30.92]},
        {"name": "acetonitrile", "h": [(2.10, "s")], "c": [116.43, 1.89]},
        {"name": "methanol", "h": [(3.49, "s") , (1.05, "s")], "c": [50.41]},
        {"name": "ethanol", "h": [(3.72, "q"), (1.24, "t"), (1.42, "s")], "c": [58.28, 18.41]},
        {"name": "ethyl acetate", "h": [(2.05, "s"), (4.12, "q"), (1.26, "t")], "c": [171.36, 60.49, 21.04, 14.19]},
        {"name": "diethyl ether", "h":[(1.21, "t"), (3.48, "q")], "c": [65.9, 15.2]},
        {"name": "n-hexane", "h": [(0.88, "t"), (1.27, "m")], "c": [14.14, 22.70, 31.64]},
        {"name": "cyclohexane", "h": [(1.43, "s")], "c": [26.94]},
        {"name": "MTBE", "h": [(3.22, "s"), (1.19, "s")], "c": [72.87, 49.45, 26.99]},
        {"name": "dichloromethane", "h": [(5.30, "s")], "c": [53.52]},
        {"name": "DMSO", "h": [(2.62, "s")], "c": [40.76]},
        {"name": "THF", "h": [(3.75, "m"), (1.85, "m")], "c": [68.00, 25.68]},
    ],
    "dmso-d6": [
        {"name": "DMSO-d6 residual", "h": [(2.50, "s")], "c": [39.53]},
        {"name": "water", "h": [(3.33, "s")], "c": []}, # water can only be detected by H-peak
        {"name": "acetone", "h": [(2.09, "s")], "c": [206.31, 30.56]},
        {"name": "acetonitrile", "h": [(2.07, "s")], "c": [117.91, 1.03]},
        {"name": "methanol", "h": [(3.17, "d"), (4.10, "q")], "c": [48.59]},
        {"name": "ethanol", "h": [(3.44, "qd"), (1.06, "t"), (4.35, "t")], "c": [56.07, 18.51]},
        {"name": "ethanol-2", "h": [(3.44, "m"), (1.06, "t"), (4.35, "t")], "c": [56.07, 18.51]},
        {"name": "ethyl acetate", "h": [(1.99, "s"), (4.03, "q"), (1.17, "t")], "c": [170.31, 59.74, 20.68, 14.40]},
        {"name": "diethyl ether", "h": [(1.09, "t"), (3.38, "q")], "c": [15.12, 62.05]},
        {"name": "n-hexane", "h": [(0.86, "t"), (1.25, "m")], "c": [13.88, 22.05, 30.95]},
        {"name": "cyclohexane", "h": [(1.40, "s")], "c": [26.33]},
        {"name": "MTBE", "h": [(3.08, "s"), (1.11, "s")], "c": [72.04, 48.70, 26.79]},
        {"name": "dichloromethane", "h": [(5.76, "s")], "c": [54.84]},
        {"name": "CHCl3", "h": [(7.26, "s")], "c": [77.06]},
        {"name": "THF", "h": [(3.605, "m"), (1.765, "m")], "c": [67.07, 25.19]},
    ],
}


def get_solvent_library_by_nmr_solvent() -> Dict[str, List[SolventPeak]]:
    lib: Dict[str, List[SolventPeak]] = {}

    for k, items in SOLVENT_LIBRARY_BY_NMR_SOLVENT.items():
        peaks: List[SolventPeak] = []
        for it in items:
            name = it.get("name", "")
            h = it.get("h", []) or []
            c = it.get("c", []) or []
            peaks.append(SolventPeak(name=name, h_peaks=h, c_peaks=c))
        lib[k] = peaks

    return lib


# main function
def detect_solvents_in_spectrum_by_nmr_solvent(
    h_shifts: List[float],
    h_mults: List[str],
    c_shifts: List[float],
    nmr_solvent: str,
    library_by_solvent: Optional[Dict[str, List[SolventPeak]]] = None,
    tol_h: float = 0.02,
    tol_c: float = 0.2,
) -> Dict[str, Any]:

    lib = library_by_solvent or get_solvent_library_by_nmr_solvent()
    
    try:
        candidates = lib[nmr_solvent]
    except KeyError:
        raise ValueError(f"No solvent library entries found for NMR solvent '{nmr_solvent}'.")

    user_h = list(zip(h_shifts, h_mults))
    user_c = list(c_shifts)

    definite, possible, details = {}, {}, {}

    for sol in candidates:
        h_total = len(sol.h_peaks)
        c_total = len(sol.c_peaks)

        h_matched, h_map = (_assign_matches_h(sol.h_peaks, user_h, tol_h) if h_total else (0, []))
        c_matched, c_map = (_assign_matches_1d(sol.c_peaks, user_c, tol_c) if c_total else (0, []))

        if (h_matched == h_total) and (c_matched == c_total):
            definite[sol.name] = {
                "H": [sol.h_peaks[i] for i, j in enumerate(h_map) if j is not None],
                "C": [sol.c_peaks[i] for i, j in enumerate(c_map) if j is not None],
            }
        elif (h_matched > 0) or (c_matched > 0):
            possible[sol.name] = {
                "H": [sol.h_peaks[i] for i, j in enumerate(h_map) if j is not None],
                "C": [sol.c_peaks[i] for i, j in enumerate(c_map) if j is not None],
            }

        details[sol.name] = {
            "H": {"matched": h_matched, "total": h_total, "map": h_map},
            "C": {"matched": c_matched, "total": c_total, "map": c_map},
        }

    return {"definite": definite, "possible": possible, "details": details}



h_shifts = [7.36, 1.56, 4.12, 1.80, 1.32]
h_mults  = ["s",  "s",  "s",  "s",  "s"]
c_shifts = [77.0, 31.3, 33.5, 32.0]
nmr_solvent = "cdcl3"

out = detect_solvents_in_spectrum_by_nmr_solvent(
    h_shifts, h_mults, c_shifts, nmr_solvent
)
print(out)

