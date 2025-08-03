"""
analysis_plots.py

Run GA, ACO, Hybrid on a benchmark JSON and generate:
1) Convergence plots (fitness vs generation) per algorithm + combined
2) Comparison bar charts (distance, time, cost) across algorithms
3) Best route plots (PNG with matplotlib + Folium HTML map) per algorithm

Usage:
  python analysis_plots.py --data ../data/R101.json --out ../reports/plots --runs 3

Requires:
  - Your existing backend modules:
      optimizer            (optimize_routes, evaluate_with_details)
      aco_optimizer        (aco_optimize_routes)
      hybrid_optimizer     (hybrid_optimize_routes)
  - Matplotlib, Folium (for HTML maps)
"""

import os
import json
import argparse
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt

# Optional: Folium for interactive maps
try:
    import folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

# --- Import your backend modules (same dir as this script) ---
from optimizer import optimize_routes, evaluate_with_details
from aco_optimizer import aco_optimize_routes
from hybrid_optimizer import hybrid_optimize_routes


# ---------------------------
# Helpers: data I/O and utils
# ---------------------------
def load_benchmark(filepath, depot_lat=None, depot_lon=None, assume_first_is_depot=False):
    """
    Supports multiple input shapes:
      A) {"base": {...}, "points": [...]}
      B) [ {...}, {...}, ... ]  (plain list of stops, no depot)
      C) [ {...}, {...}, ... ]  (plain list including depot with id==0)
    Optional:
      - provide depot_lat/depot_lon to force a depot
      - assume_first_is_depot=True to treat first element as the depot
    """
    with open(filepath, "r") as f:
        raw = json.load(f)

    # Case A: project format
    if isinstance(raw, dict) and "base" in raw and "points" in raw:
        base = SimpleNamespace(**raw["base"])
        points = [SimpleNamespace(**p) for p in raw["points"]]
        return base, points

    # Case B/C: plain list of dicts
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        items = raw

        # If a depot is explicitly present (id == 0), use it
        depot = None
        for p in items:
            if str(p.get("id", "")).strip() == "0":
                depot = p
                break

        # If not present, derive depot
        if depot is None:
            if depot_lat is not None and depot_lon is not None:
                depot = {"id": 0, "lat": float(depot_lat), "lon": float(depot_lon)}
            elif assume_first_is_depot:
                first = items[0]
                depot = {"id": 0, "lat": float(first["lat"]), "lon": float(first["lon"])}
                # remaining become points
                items = items[1:]
            else:
                # centroid depot
                lat = sum(float(p["lat"]) for p in items) / len(items)
                lon = sum(float(p["lon"]) for p in items) / len(items)
                depot = {"id": 0, "lat": lat, "lon": lon}

        # Build base + points (ensure unique ids)
        base = SimpleNamespace(**depot)

        points = []
        used_ids = {0}
        next_id = 1
        for p in items:
            pid = p.get("id")
            if pid is None or pid in used_ids:
                pid = next_id
                next_id += 1
            used_ids.add(pid)

            points.append(SimpleNamespace(
                id=pid,
                lat=float(p["lat"]),
                lon=float(p["lon"]),
                time_window=p.get("time_window"),
                label=p.get("label")
            ))
        return base, points

    raise ValueError("Unsupported dataset format. Expect dict with 'base' and 'points', or a list of point dicts.")

def make_point_map(base, points):
    return {p.id: vars(p) for p in [base] + points + [base]}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Evaluation normalization
# ---------------------------
def _safe_evaluate(route, point_map):
    """
    Normalize evaluate_with_details(route, point_map) to always return:
      (fitness, breakdown, metrics)
    Supports backends that return:
      - (fitness, breakdown)
      - (fitness, breakdown, metrics)
      - {"fitness":..., "breakdown":[...], "metrics":{...}} (or "route_breakdown")
    """
    out = evaluate_with_details(route, point_map)

    # Dict-style return
    if isinstance(out, dict):
        fitness = out.get("fitness")
        breakdown = out.get("breakdown") or out.get("route_breakdown") or []
        metrics = out.get("metrics")
        if metrics is None and breakdown:
            metrics = {
                "distance_km": sum(seg.get("distance_km", 0) for seg in breakdown),
                "travel_time_sec": sum(seg.get("travel_time_sec", 0) for seg in breakdown),
                "cost": sum(seg.get("segment_cost", 0) for seg in breakdown),
            }
        return fitness, breakdown, metrics

    # Tuple-style return
    if isinstance(out, tuple):
        if len(out) >= 3:
            fitness, breakdown, metrics = out[0], out[1], out[2]
            return fitness, breakdown, metrics
        elif len(out) == 2:
            fitness, breakdown = out
            metrics = {
                "distance_km": sum(seg.get("distance_km", 0) for seg in breakdown),
                "travel_time_sec": sum(seg.get("travel_time_sec", 0) for seg in breakdown),
                "cost": sum(seg.get("segment_cost", 0) for seg in breakdown),
            }
            return fitness, breakdown, metrics

    # Unknown shape
    raise ValueError("evaluate_with_details returned an unsupported shape")


def normalize_run_result(label, run_output, point_map, rate_km=0.25, rate_hour=25.0, force_derive=False):
    """
    Normalize solver output (dict or tuple) to a standard dict and
    ALWAYS produce distance_km, travel_time_sec, cost derived from breakdown,
    ignoring backend-provided cost to avoid unit inconsistencies.
    """
    result = {
        "label": label,
        "best_route": None,
        "best_breakdown": None,
        "best_metrics": None,
        "routes_over_time": [],
        "fitness_over_time": []
    }

    # Dict-like
    if isinstance(run_output, dict):
        result["best_route"] = run_output.get("best_route")
        result["routes_over_time"] = run_output.get("routes_over_time") or []
        result["fitness_over_time"] = run_output.get("fitness_over_time") or []

        provided_breakdown = run_output.get("route_breakdown") or run_output.get("breakdown")

        if provided_breakdown:
            result["best_breakdown"] = provided_breakdown
            result["best_metrics"] = _aggregate_metrics_from_breakdown(
                provided_breakdown, rate_km=rate_km, rate_hour=rate_hour
            )
        else:
            # No breakdown given: recompute if we have a route
            if result["best_route"]:
                _, brk, _ = _safe_evaluate(result["best_route"], point_map)
                result["best_breakdown"] = brk
                result["best_metrics"] = _aggregate_metrics_from_breakdown(
                    brk, rate_km=rate_km, rate_hour=rate_hour
                )

        # If still no route, try last from routes_over_time
        if not result["best_route"] and result["routes_over_time"]:
            result["best_route"] = result["routes_over_time"][-1]
            if result["best_route"] and not result["best_breakdown"]:
                _, brk, _ = _safe_evaluate(result["best_route"], point_map)
                result["best_breakdown"] = brk
                result["best_metrics"] = _aggregate_metrics_from_breakdown(
                    brk, rate_km=rate_km, rate_hour=rate_hour
                )
        return result

    # Tuple-like (route, score, ...)
    if isinstance(run_output, (tuple, list)) and len(run_output) >= 1:
        route = run_output[0]
        result["best_route"] = route
        if route:
            _, brk, _ = _safe_evaluate(route, point_map)
            result["best_breakdown"] = brk
            result["best_metrics"] = _aggregate_metrics_from_breakdown(
                brk, rate_km=rate_km, rate_hour=rate_hour
            )
        return result

    return result




def _compute_fitness_series_from_routes(routes_over_time, point_map):
    """If a solver didn't return fitness_over_time, compute it from routes_over_time."""
    series = []
    for rt in routes_over_time:
        if not rt:
            series.append(float('inf'))
            continue
        fit, _, _ = _safe_evaluate(rt, point_map)
        series.append(fit if fit is not None else float('inf'))
    return series


# ---------------------------
# Running the three algorithms
# ---------------------------
def run_algorithms(base, points, point_map, seed=None):
    """
    Run GA, ACO, Hybrid once each and normalize outputs.
    GA: no point_map
    ACO/Hybrid: require point_map and return_history=True
    """
    if seed is not None:
        random.seed(seed)

    outputs = []

    # --- GA (dict in your repo, no point_map) ---
    ga_out = optimize_routes(points=points, base_point=base)
    ga_res = normalize_run_result("GA", ga_out, point_map)
    outputs.append(ga_res)

    # --- ACO (6-tuple when return_history=True) ---
    aco_out = aco_optimize_routes(points=points, base_point=base, point_map=point_map, return_history=True)
    if isinstance(aco_out, (tuple, list)) and len(aco_out) >= 6:
        best_route, best_cost, routes_over_time, fitness_over_time, pheromone_history, metrics_over_time = aco_out
        aco_res_input = {
            "best_route": best_route,
            "routes_over_time": routes_over_time or [],
            "fitness_over_time": fitness_over_time or [],
            "metrics": (metrics_over_time[-1] if metrics_over_time else None),
        }
        aco_res = normalize_run_result("ACO", aco_res_input, point_map)
        # derive fitness if missing but routes exist
        if not aco_res.get("fitness_over_time") and aco_res.get("routes_over_time"):
            aco_res["fitness_over_time"] = _compute_fitness_series_from_routes(aco_res["routes_over_time"], point_map)
        outputs.append(aco_res)
    else:
        raise ValueError("ACO output format unexpected. Expected 6-tuple with return_history=True")

    # --- Hybrid (5-tuple when return_history=True) ---
    hyb_out = hybrid_optimize_routes(points=points, base_point=base, point_map=point_map, return_history=True)
    if isinstance(hyb_out, (tuple, list)) and len(hyb_out) >= 5:
        best_route, best_score, routes_over_time, fitness_over_time, metrics_over_time = hyb_out
        hyb_res_input = {
            "best_route": best_route,
            "routes_over_time": routes_over_time or [],
            "fitness_over_time": fitness_over_time or [],
            "metrics": (metrics_over_time[-1] if metrics_over_time else None),
        }
        hyb_res = normalize_run_result("Hybrid", hyb_res_input, point_map)
        if not hyb_res.get("fitness_over_time") and hyb_res.get("routes_over_time"):
            hyb_res["fitness_over_time"] = _compute_fitness_series_from_routes(hyb_res["routes_over_time"], point_map)
        outputs.append(hyb_res)
    else:
        raise ValueError("Hybrid output format unexpected. Expected 5-tuple with return_history=True")

    # FINAL SAFETY: ensure best_metrics exists for all
    for res in outputs:
        if res.get("best_route") and not res.get("best_metrics"):
            _, brk, mets = _safe_evaluate(res["best_route"], point_map)
            res["best_breakdown"] = res.get("best_breakdown") or brk
            res["best_metrics"] = mets

    return outputs


# ---------------------------
# Plotting helpers (matplotlib)
# ---------------------------
def plot_convergence(all_runs, out_dir, dataset_name):
    """
    Also saves:
      - convergence_<dataset>.csv            (wide)
      - convergence_<dataset>_<ALG>.csv     (long)
    """
    ensure_dir(out_dir)

    # Build a unified table in memory
    # Find max length per algorithm for padding
    algo_series = {}
    max_len = 0
    for res in all_runs:
        fos = res.get("fitness_over_time") or []
        if fos:
            algo_series[res["label"]] = list(fos)
            max_len = max(max_len, len(fos))

    # Save per-algorithm long CSVs
    for label, series in algo_series.items():
        rows = [(i, float(v)) for i, v in enumerate(series)]
        _save_dataframe_csv(rows, ["generation", "fitness"], os.path.join(out_dir, f"convergence_{dataset_name}_{label}.csv"))

    # Save combined wide CSV
    if algo_series:
        generations = list(range(max_len))
        # pad with last value
        def pad(seq, L):
            if not seq: return ["" for _ in range(L)]
            if len(seq) < L: seq = seq + [seq[-1]] * (L - len(seq))
            return seq
        rows = []
        header = ["generation"] + list(algo_series.keys())
        for g in generations:
            row = [g]
            for label in algo_series.keys():
                seq = pad(algo_series[label], max_len)
                row.append(float(seq[g]))
            rows.append(row)
        _save_dataframe_csv(rows, header, os.path.join(out_dir, f"convergence_{dataset_name}.csv"))

    # Existing plotting logic (unchanged)
    plt.figure(figsize=(8, 4))
    any_series = False
    for label, series in algo_series.items():
        if series:
            plt.plot(range(len(series)), series, label=label)
            any_series = True
    if any_series:
        plt.xlabel("Generation")
        plt.ylabel("Fitness (lower is better)")
        plt.title(f"Convergence: {dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"convergence_{dataset_name}.png"))
        plt.savefig(os.path.join(out_dir, f"convergence_{dataset_name}.pdf"))
    plt.close()

    # Per-algo plots
    for label, series in algo_series.items():
        if not series:
            continue
        plt.figure(figsize=(7, 3.5))
        plt.plot(range(len(series)), series)
        plt.xlabel("Generation")
        plt.ylabel("Fitness (lower is better)")
        plt.title(f"Convergence: {dataset_name} - {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"convergence_{dataset_name}_{label}.png"))
        plt.savefig(os.path.join(out_dir, f"convergence_{dataset_name}_{label}.pdf"))
        plt.close()



def plot_comparison_bars(all_runs, out_dir, dataset_name):
    """
    Also saves:
      - compare_metrics_<dataset>.csv
    """
    ensure_dir(out_dir)

    labels, distances, times_min, costs = [], [], [], []
    for res in all_runs:
        labels.append(res["label"])
        m = res.get("best_metrics") or {"distance_km": 0.0, "travel_time_sec": 0.0, "cost": 0.0}
        d = float(m.get("distance_km", 0.0))
        t_sec = float(m.get("travel_time_sec", 0.0))
        c = float(m.get("cost", 0.0))
        distances.append(d)
        times_min.append(t_sec / 60.0)
        costs.append(c)

    # Save CSV
    rows = []
    for i, label in enumerate(labels):
        rows.append([label, distances[i], times_min[i] * 60.0, times_min[i], costs[i]])
    _save_dataframe_csv(
        rows,
        ["algorithm", "distance_km", "travel_time_sec", "travel_time_min", "cost"],
        os.path.join(out_dir, f"compare_metrics_{dataset_name}.csv")
    )

    # Plot
    x = list(range(len(labels)))
    width = 0.25
    plt.figure(figsize=(9, 4))
    plt.bar([i - width for i in x], distances, width=width, label="Distance (km)")
    plt.bar(x, times_min, width=width, label="Time (min)")
    plt.bar([i + width for i in x], costs, width=width, label="Cost")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title(f"GA vs ACO vs Hybrid — {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"compare_metrics_{dataset_name}.png"))
    plt.savefig(os.path.join(out_dir, f"compare_metrics_{dataset_name}.pdf"))
    plt.close()





def plot_best_route_png(res, point_map, out_dir, dataset_name):
    """
    Static PNG route figure using matplotlib.
    - Numbers each node by visit order (0..N-1)
    - Highlights depot (id==0) with a star marker
    Saves: best_route_<dataset>_<label>.png / .pdf
    """
    ensure_dir(out_dir)
    route = res.get("best_route") or []
    if not route:
        return

    coords = [(point_map[pid]["lat"], point_map[pid]["lon"]) for pid in route if pid in point_map]
    if not coords:
        return

    lats = [lat for lat, _ in coords]
    lons = [lon for _, lon in coords]

    plt.figure(figsize=(7, 7))
    # Path
    plt.plot(lons, lats, linewidth=2, alpha=0.9)

    # Nodes with numeric labels
    plt.scatter(lons, lats, s=40)
    for idx, (x, y) in enumerate(zip(lons, lats)):
        plt.text(x, y, str(idx), fontsize=9, ha="center", va="bottom")

    # Start / End emphasis
    plt.scatter([lons[0]], [lats[0]], s=80, marker="o")     # Start
    plt.scatter([lons[-1]], [lats[-1]], s=80, marker="o")   # End

    # Depot (id==0) star marker wherever it appears in the route
    depot_positions = [i for i, pid in enumerate(route) if pid == 0]
    for i in depot_positions:
        plt.scatter([lons[i]], [lats[i]], s=120, marker="*", edgecolor="k")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Best Route — {dataset_name} — {res['label']} (numbers = visit order)")
    plt.tight_layout()
    fname = f"best_route_{dataset_name}_{res['label']}"
    plt.savefig(os.path.join(out_dir, f"{fname}.png"))
    plt.savefig(os.path.join(out_dir, f"{fname}.pdf"))
    plt.close()



def plot_best_route_folium(res, point_map, out_dir, dataset_name):
    """
    Folium HTML interactive map for the best route.
    - Numbers each node (order)
    - Green home icon for depot (id==0)
    - Start/End colors
    Saves: best_route_<dataset>_<label>.html
    """
    if not FOLIUM_OK:
        return

    ensure_dir(out_dir)
    route = res.get("best_route") or []
    if not route:
        return

    coords = [(point_map[pid]["lat"], point_map[pid]["lon"]) for pid in route if pid in point_map]
    if not coords:
        return

    m = folium.Map(location=coords[0], zoom_start=13)

    for idx, pid in enumerate(route):
        if pid not in point_map:
            continue
        lat, lon = point_map[pid]["lat"], point_map[pid]["lon"]

        # Choose icon: depot vs regular
        if pid == 0:
            icon = folium.Icon(color="darkpurple", icon="home", prefix="fa")
            label = f"Depot (order {idx})"
        else:
            # start / end / middle
            if idx == 0:
                icon = folium.Icon(color="green", icon="flag", prefix="fa")
            elif idx == len(route) - 1:
                icon = folium.Icon(color="red", icon="flag-checkered", prefix="fa")
            else:
                icon = folium.Icon(color="blue", icon="info-sign")
            label = f"Stop {idx} (ID {pid})"

        # Popup with numbering
        popup = folium.Popup(f"<b>{label}</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}", max_width=300)
        # Add a small number overlay using a DivIcon
        number_icon = folium.DivIcon(html=f"""
            <div style="font-size: 10px; color: black; background: white; border-radius: 8px; padding: 1px 3px; border:1px solid #444;">
                {idx}
            </div>
        """)

        # Base icon marker
        folium.Marker(location=(lat, lon), icon=icon, popup=popup, tooltip=label).add_to(m)
        # Number overlay (slightly offset by adding another marker)
        folium.Marker(location=(lat, lon), icon=number_icon).add_to(m)

    folium.PolyLine(coords, color="blue", weight=5, opacity=0.85).add_to(m)
    out_html = os.path.join(out_dir, f"best_route_{dataset_name}_{res['label']}.html")
    m.save(out_html)


def _aggregate_metrics_from_breakdown(breakdown, rate_km=0.25, rate_hour=25.0):
    """
    Aggregate distance (km) and time (sec) from a list of per-segment dicts
    using consistent rules, then derive a monetary cost:
        cost = rate_km * distance_km + rate_hour * (time_sec / 3600).
    We intentionally IGNORE any 'segment_cost'/'total_cost' provided by the backend
    to avoid unit inconsistencies across algorithms.
    """
    total_dist = 0.0
    total_time = 0.0

    for seg in breakdown or []:
        # Distance (km)
        total_dist += float(
            seg.get("distance_km")
            or seg.get("segment_distance_km")
            or seg.get("distance", 0.0)
        )

        # Time (sec) — prefer consolidated if present, else sum components
        t = seg.get("travel_time_sec") or seg.get("total_time_sec") or seg.get("time_sec")
        if t is not None:
            total_time += float(t)
        else:
            base = float(seg.get("base_time_sec", 0.0))
            traf = float(seg.get("traffic_penalty_sec", 0.0))
            wx   = float(seg.get("weather_penalty_sec", 0.0))
            tw   = float(seg.get("time_window_penalty_sec", 0.0))
            total_time += base + traf + wx + tw

    # Derive a consistent cost (ignore backend cost fields)
    total_cost = rate_km * total_dist + rate_hour * (total_time / 3600.0)

    return {
        "distance_km": total_dist,
        "travel_time_sec": total_time,
        "cost": total_cost
    }


def _save_dataframe_csv(rows, columns, path):
    import csv
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

def save_best_route_csvs(res, point_map, out_dir, dataset_name):
    """
    Saves:
      - best_route_<dataset>_<ALG>.csv  (order, stop_id, lat, lon)
      - best_route_breakdown_<dataset>_<ALG>.csv  (whatever breakdown fields exist)
    """
    ensure_dir(out_dir)
    label = res["label"]
    route = res.get("best_route") or []
    if route:
        rows = []
        for idx, pid in enumerate(route):
            pt = point_map.get(pid, {})
            rows.append([idx, pid, pt.get("lat", ""), pt.get("lon", "")])
        _save_dataframe_csv(rows, ["order", "stop_id", "lat", "lon"],
                            os.path.join(out_dir, f"best_route_{dataset_name}_{label}.csv"))

    brk = res.get("best_breakdown") or []
    if brk:
        # Extract headers union across all segments
        headers = set()
        for seg in brk:
            headers |= set(seg.keys())
        headers = list(headers)
        # Write rows in that header order
        rows = [[seg.get(h, "") for h in headers] for seg in brk]
        _save_dataframe_csv(rows, headers, os.path.join(out_dir, f"best_route_breakdown_{dataset_name}_{label}.csv"))


def plot_best_routes_combined_png(all_runs, point_map, out_dir, dataset_name):
    """
    Overlay GA, ACO, Hybrid best routes on one static plot with numbering and depot mark.
    Saves: best_routes_combined_<dataset>.png/.pdf
    """
    ensure_dir(out_dir)

    curves = []  # (label, lons, lats, route_ids)
    for res in all_runs:
        route = res.get("best_route") or []
        if not route:
            continue
        coords = [(point_map[pid]["lat"], point_map[pid]["lon"]) for pid in route if pid in point_map]
        if not coords:
            continue
        lats = [lat for lat, _ in coords]
        lons = [lon for _, lon in coords]
        curves.append((res["label"], lons, lats, route))

    if not curves:
        return

    plt.figure(figsize=(7.5, 7.5))
    styles = {
        "GA":     {"linewidth": 2.0, "marker": "o"},
        "ACO":    {"linewidth": 2.0, "marker": "s"},
        "Hybrid": {"linewidth": 2.0, "marker": "D"},
    }

    for label, lons, lats, route in curves:
        stl = styles.get(label, {"linewidth": 2.0})
        plt.plot(lons, lats, label=label, **stl)
        plt.scatter(lons, lats, s=35)

        # Numbering
        for idx, (x, y) in enumerate(zip(lons, lats)):
            plt.text(x, y, str(idx), fontsize=8, ha="center", va="bottom")

        # Depot star(s)
        for i, pid in enumerate(route):
            if pid == 0:
                plt.scatter([lons[i]], [lats[i]], s=120, marker="*", edgecolor="k")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Best Routes (Overlay) — {dataset_name} (numbers = visit order)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    out = os.path.join(out_dir, f"best_routes_combined_{dataset_name}")
    plt.savefig(out + ".png")
    plt.savefig(out + ".pdf")
    plt.close()



def save_combined_routes_csv(all_runs, point_map, out_dir, dataset_name):
    """
    Saves best_routes_combined_<dataset>.csv with columns:
    algorithm, order, stop_id, lat, lon
    """
    ensure_dir(out_dir)
    rows = []
    for res in all_runs:
        route = res.get("best_route") or []
        for idx, pid in enumerate(route):
            pt = point_map.get(pid, {})
            rows.append([res["label"], idx, pid, pt.get("lat", ""), pt.get("lon", "")])
    _save_dataframe_csv(
        rows,
        ["algorithm", "order", "stop_id", "lat", "lon"],
        os.path.join(out_dir, f"best_routes_combined_{dataset_name}.csv")
    )


def plot_best_routes_combined_folium(all_runs, point_map, out_dir, dataset_name):
    """
    Single interactive Folium map with each algorithm's best route in its own layer.
    Saves: best_routes_combined_<dataset>.html
    """
    if not FOLIUM_OK:
        return
    ensure_dir(out_dir)

    # Find a center point from the first available route
    center = None
    for res in all_runs:
        route = res.get("best_route") or []
        if route:
            pid0 = route[0]
            if pid0 in point_map:
                center = (point_map[pid0]["lat"], point_map[pid0]["lon"])
                break
    if center is None:
        return

    m = folium.Map(location=center, zoom_start=13)

    # Distinct colors per algorithm
    colors = {"GA": "blue", "ACO": "green", "Hybrid": "red"}

    for res in all_runs:
        label = res["label"]
        route = res.get("best_route") or []
        if not route:
            continue
        coords = [(point_map[pid]["lat"], point_map[pid]["lon"]) for pid in route if pid in point_map]
        if not coords:
            continue

        layer = folium.FeatureGroup(name=f"{label} best route", show=True)

        # Markers
        for idx, (lat, lon) in enumerate(coords):
            tip = f"{label}: Stop {idx} (ID {route[idx]})"
            icon_color = "blue"
            if idx == 0:
                icon_color = "green"
                tip += " (Start)"
            elif idx == len(coords) - 1:
                icon_color = "red"
                tip += " (End)"
            folium.Marker(
                (lat, lon),
                tooltip=tip,
                icon=folium.Icon(color=icon_color, icon="info-sign")
            ).add_to(layer)

        # Polyline for this algorithm
        folium.PolyLine(
            coords,
            color=colors.get(label, "blue"),
            weight=5,
            opacity=0.85
        ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html = os.path.join(out_dir, f"best_routes_combined_{dataset_name}.html")
    m.save(out_html)


def plot_best_routes_grid_png(all_runs, point_map, out_dir, dataset_name, ncols=3):
    """
    Draw GA, ACO, Hybrid best routes as separate subplots in one figure.
    - Consistent axis limits across panels (so shapes are comparable)
    - Numbers each node by visit order (0..N-1)
    - Marks depot (id==0) with a star
    Saves: best_routes_grid_<dataset>.png/.pdf
    """
    ensure_dir(out_dir)

    # Collect data
    panels = []  # list of dicts: {"label":..., "lons":[...], "lats":[...], "route":[...]}
    all_lons, all_lats = [], []
    for res in all_runs:
        route = res.get("best_route") or []
        coords = [(point_map[pid]["lat"], point_map[pid]["lon"]) for pid in route if pid in point_map]
        if not route or not coords:
            continue
        lats = [lat for lat, _ in coords]
        lons = [lon for _, lon in coords]
        panels.append({"label": res["label"], "lons": lons, "lats": lats, "route": route})
        all_lons.extend(lons)
        all_lats.extend(lats)

    if not panels:
        return

    # Determine grid (default 1x3; fall back to 2x2 if needed)
    n = len(panels)
    if ncols <= 0:
        ncols = 3
    nrows = (n + ncols - 1) // ncols

    # Global bounds for consistent axes
    margin = 0.01
    xmin, xmax = min(all_lons) - margin, max(all_lons) + margin
    ymin, ymax = min(all_lats) - margin, max(all_lats) + margin

    import matplotlib.pyplot as plt
    figsize = (5 * ncols, 5 * nrows)  # scale with grid size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    # Basic style per algorithm
    styles = {
        "GA":     {"linewidth": 2.0, "marker": "o"},
        "ACO":    {"linewidth": 2.0, "marker": "s"},
        "Hybrid": {"linewidth": 2.0, "marker": "D"},
    }

    # Draw each panel
    idx_panel = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.2, linestyle="--")

            if idx_panel < n:
                p = panels[idx_panel]
                label = p["label"]
                lons, lats, route = p["lons"], p["lats"], p["route"]

                stl = styles.get(label, {"linewidth": 2.0})
                ax.plot(lons, lats, **stl)
                ax.scatter(lons, lats, s=35)

                # Numbering
                for i, (x, y) in enumerate(zip(lons, lats)):
                    ax.text(x, y, str(i), fontsize=9, ha="center", va="bottom")

                # Depot star(s) where id==0 appears in the route
                for i, pid in enumerate(route):
                    if pid == 0:
                        ax.scatter([lons[i]], [lats[i]], s=120, marker="*", edgecolor="k")

                ax.set_title(f"{label} best route")
            else:
                # Empty panel if fewer algos than grid cells
                ax.axis("off")

            idx_panel += 1

    fig.suptitle(f"Best Routes — {dataset_name} (numbers = visit order, ★=depot)", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(out_dir, f"best_routes_grid_{dataset_name}")
    plt.savefig(out + ".png", dpi=200)
    plt.savefig(out + ".pdf")
    plt.close(fig)



# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate convergence, comparison, and route plots for GA/ACO/Hybrid")
    parser.add_argument("--data", required=True, help="Path to benchmark JSON (with 'base' and 'points' OR list of points)")
    parser.add_argument("--out", default="plots", help="Output directory for plots")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per algorithm (seeds)")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (optional)")
    parser.add_argument("--depot-lat", type=float, default=None, help="Override depot latitude if data has no depot")
    parser.add_argument("--depot-lon", type=float, default=None, help="Override depot longitude if data has no depot")
    parser.add_argument("--first-is-depot", action="store_true", help="Treat the first list element as the depot")
    parser.add_argument("--rate-km", type=float, default=0.25, help="Cost rate per km")
    parser.add_argument("--rate-hour", type=float, default=25.0, help="Cost rate per hour")
    parser.add_argument("--force-derive-metrics", action="store_true",
                        help="Ignore backend metrics and always derive from breakdown")

    args = parser.parse_args()

    ensure_dir(args.out)
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    base, points = load_benchmark(args.data, depot_lat=args.depot_lat, depot_lon=args.depot_lon, assume_first_is_depot=args.first_is_depot)
    point_map = make_point_map(base, points)

    # Aggregate results across runs: keep the best-by-fitness per algorithm,
    # and average convergence if multiple runs are available.
    aggregated = {
        "GA": {"label": "GA", "best_route": None, "best_breakdown": None, "best_metrics": None, "fitness_over_time_runs": []},
        "ACO": {"label": "ACO", "best_route": None, "best_breakdown": None, "best_metrics": None, "fitness_over_time_runs": []},
        "Hybrid": {"label": "Hybrid", "best_route": None, "best_breakdown": None, "best_metrics": None, "fitness_over_time_runs": []},
    }

    # For each run, execute the three algorithms once
    for r in range(args.runs):
        seed_r = (args.seed + r) if args.seed is not None else None
        res_list = run_algorithms(base, points, point_map, seed=seed_r)

        for res in res_list:
            label = res["label"]
            # Keep per-run fitness series (for averaging)
            run_fot = res.get("fitness_over_time") or []
            if run_fot:
                aggregated[label]["fitness_over_time_runs"].append(run_fot)

            # Choose "best" result per label as the one with minimal final fitness if histories exist, else first non-empty
            def final_fit(arr):
                return arr[-1] if arr else float("inf")

            cur_best = aggregated[label]["best_route"]
            cur_best_fit = float("inf")
            if cur_best:
                kept_runs = aggregated[label]["fitness_over_time_runs"]
                if kept_runs:
                    cur_best_fit = min(final_fit(x) for x in kept_runs)

            new_fit = final_fit(run_fot)
            take_it = False
            if not cur_best and res.get("best_route"):
                take_it = True
            elif new_fit < cur_best_fit:
                take_it = True

            if take_it and res.get("best_route"):
                aggregated[label]["best_route"] = res["best_route"]
                aggregated[label]["best_breakdown"] = res.get("best_breakdown")
                aggregated[label]["best_metrics"] = res.get("best_metrics")

    # Build combined per-algorithm objects for plotting
    final_results = []
    for label, blob in aggregated.items():
        # Average the fitness_over_time across runs, if available
        fot_runs = blob["fitness_over_time_runs"]
        if fot_runs:
            # Pad sequences to same length with last value
            max_len = max(len(x) for x in fot_runs)
            padded = []
            for seq in fot_runs:
                if len(seq) < max_len:
                    seq = seq + [seq[-1]] * (max_len - len(seq))
                padded.append(seq)
            avg = [sum(vals)/len(vals) for vals in zip(*padded)]
        else:
            avg = []

        final_results.append({
            "label": label,
            "best_route": blob["best_route"],
            "best_breakdown": blob["best_breakdown"],
            "best_metrics": blob["best_metrics"],
            "fitness_over_time": avg
        })

    # 1) Convergence plots
    plot_convergence(final_results, args.out, dataset_name)

    # 2) Comparison bars (distance/time/cost)
    plot_comparison_bars(final_results, args.out, dataset_name)

    # 3) Best route plots (PNG + optional HTML)
    for res in final_results:
        save_best_route_csvs(res, point_map, args.out, dataset_name)  # <-- add this
        plot_best_route_png(res, point_map, args.out, dataset_name)
        plot_best_route_folium(res, point_map, args.out, dataset_name)

    # Combined best routes overlay (PNG/PDF)
    plot_best_routes_combined_png(final_results, point_map, args.out, dataset_name)
    plot_best_routes_combined_folium(final_results, point_map, args.out, dataset_name)  # if you added this
    save_combined_routes_csv(final_results, point_map, args.out, dataset_name)  # <-- add this
    # Combined best-route grid (separate subplots)
    plot_best_routes_grid_png(final_results, point_map, args.out, dataset_name, ncols=3)

    print(f"✅ Done. Plots written to: {os.path.abspath(args.out)}")
    for fn in sorted(os.listdir(args.out)):
        print(" -", fn)


if __name__ == "__main__":
    main()
