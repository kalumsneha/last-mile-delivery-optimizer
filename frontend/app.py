import streamlit as st
import requests
import json
import pandas as pd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import AntPath, MarkerCluster
import matplotlib.ticker as ticker

st.set_page_config(layout="wide")
st.title("🚚 Last-Mile Delivery Optimizer (Enhanced)")

base_point = {"id": 0, "lat": 43.651070, "lon": -79.347015, "time_window": None, "label": "Depot (Base)"}

def parse_solomon_file(file_obj):
    customers_started = False
    points = []
    center_lat = 43.651070
    center_lon = -79.347015
    x_scale = 0.02
    y_scale = 0.02

    for line in file_obj:
        decoded = line.decode("utf-8").strip()
        if not customers_started:
            if decoded.startswith("CUST NO."):
                customers_started = True
                next(file_obj)
                continue
        else:
            tokens = decoded.split()
            if len(tokens) < 7:
                continue

            cust_id = int(tokens[0])
            if cust_id == 0:
                continue
            x = float(tokens[1])
            y = float(tokens[2])
            ready = int(tokens[4])
            due = int(tokens[5])
            lat = center_lat + (y - 50) * y_scale
            lon = center_lon + (x - 50) * x_scale
            points.append({
                "id": cust_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "label": f"Customer {cust_id}",
                "time_window": [ready, due]
            })
    return points

uploaded = st.file_uploader("Upload delivery points (.json or Solomon .txt)", type=["json", "txt"])

if uploaded and "uploaded_points" not in st.session_state:
    filename = uploaded.name.lower()
    if filename.endswith(".json"):
        st.session_state["uploaded_points"] = json.load(uploaded)
    elif filename.endswith(".txt"):
        st.session_state["uploaded_points"] = parse_solomon_file(uploaded)
    else:
        st.error("Unsupported file type.")
    st.session_state["results"] = {}

if "uploaded_points" in st.session_state:
    points = st.session_state["uploaded_points"]
    initial_map = folium.Map(location=[base_point["lat"], base_point["lon"]], zoom_start=11)
    all_points = points + [base_point]

    enable_cluster = st.checkbox("Enable Marker Clustering (Initial Map)", value=True)
    marker_cluster = MarkerCluster().add_to(initial_map) if enable_cluster else None

    for p in all_points:
        try:
            lat = float(p["lat"])
            lon = float(p["lon"])
            label = p.get("label", f"Point {p['id']}")
            time_window = p.get("time_window")
        except (KeyError, TypeError, ValueError):
            continue  # skip bad data

        # 🟦 Determine color by time window
        if p["id"] == 0:
            color = "darkpurple"  # depot
        elif not time_window:
            color = "gray"
        else:
            start_hour = time_window[0]
            if start_hour < 12:
                color = "green"
            elif start_hour < 17:
                color = "orange"
            else:
                color = "red"

        icon = folium.Icon(color=color, icon="info-sign")

        popup_html = f"""
            <b>{label}</b><br>
            Lat: {lat:.6f}<br>
            Lon: {lon:.6f}<br>
        """
        if time_window:
            popup_html += f"Time Window: {time_window[0]}–{time_window[1]}"

        marker = folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{label} ({lat:.4f}, {lon:.4f})",
            icon=icon
        )

        if marker_cluster:
            marker.add_to(marker_cluster)
        else:
            marker.add_to(initial_map)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 230px;
        border: 2px solid grey;
        background-color: white;
        z-index:9999;
        font-size: 14px;
        padding: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        color: black;
    ">
    <b>🗂️ Time Window Legend</b><br>
    <span style='color:green; font-weight:bold;'>●</span> Morning (before 12 PM)<br>
    <span style='color:orange; font-weight:bold;'>●</span> Afternoon (12–17 PM)<br>
    <span style='color:red; font-weight:bold;'>●</span> Evening/Late (after 5 PM)<br>
    <span style='color:purple; font-weight:bold;'>●</span> Depot<br>
    <span style='color:gray; font-weight:bold;'>●</span> Unknown
    </div>
    """
    initial_map.get_root().html.add_child(folium.Element(legend_html))

    st.subheader("🗺️ Initial Delivery Points Map (Color-Coded by Time Window)")
    st_folium(initial_map, width=800, height=550)


    if st.button("Run All Optimizations"):
        st.session_state["results"] = {}
        for algo in ["GA", "ACO", "Hybrid"]:
            with st.spinner(f"Running {algo} optimization..."):
                res = requests.post(
                    "http://localhost:8000/optimize",
                    json={"points": points, "algorithm": algo}
                )
                try:
                    st.session_state["results"][algo] = res.json()
                except Exception as e:
                    st.error(f"❌ Error decoding response from {algo}: {e}")
                    st.code(res.text)
                    st.stop()

if "results" in st.session_state and st.session_state["results"]:
    algo_selected = st.selectbox("Select algorithm to view result:", list(st.session_state["results"].keys()))
    result = st.session_state["results"][algo_selected]
    points = st.session_state["uploaded_points"]
    all_points = {p["id"]: p for p in points + [base_point]}

    st.subheader(f"Optimization Result: {algo_selected}")

    routes_over_time = result.get("routes_over_time", [])
    breakdowns = result.get("route_breakdown", [])
    fitness_over_time = result.get("fitness_over_time", [])

    if routes_over_time and len(routes_over_time) > 1:
        gen_selected = st.slider(
            "Select generation",
            min_value=0,
            max_value=len(routes_over_time) - 1,
            value=len(routes_over_time) - 1
        )
        route = routes_over_time[gen_selected]
    elif routes_over_time:
        st.info("Only one generation available.")
        route = routes_over_time[0]
    elif result.get("best_route"):
        st.info("No generation history. Using best route.")
        route = result["best_route"]
    else:
        st.warning("No generation history or route available for this algorithm.")
        route = []

    if route:
        coords = [(all_points[pid]["lat"], all_points[pid]["lon"]) for pid in route]
    else:
        coords = []

    if coords:
        m = folium.Map(location=coords[0], zoom_start=13)
        enable_cluster = st.checkbox("Enable Marker Clustering", value=False)

        if enable_cluster:
            marker_cluster = MarkerCluster().add_to(m)

        for i, pid in enumerate(route):
            p = all_points[pid]
            if pid == base_point["id"]:
                icon = folium.Icon(color="green", icon="home")
                label = f"Depot (Stop {i+1})"
            elif i == 0:
                icon = folium.Icon(color="green", icon="info-sign")
                label = f"Start (Stop {i+1})"
            elif i == len(route) - 1:
                icon = folium.Icon(color="red", icon="info-sign")
                label = f"End (Stop {i+1})"
            else:
                icon = folium.Icon(color="blue", icon="info-sign")
                label = f"Stop {i+1}"

            popup_text = label
            if breakdowns and i < len(breakdowns):
                seg = breakdowns[i]
                eta = seg.get("travel_time_sec", 0) / 60
                cost = seg.get("segment_cost", seg.get("total_cost_sec", 0))
                popup_text += f"<br>ETA: {eta:.1f} min<br>Cost: ${cost:.2f}"

            marker = folium.Marker([p["lat"], p["lon"]], popup=popup_text, icon=icon)
            if enable_cluster:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)

        folium.PolyLine(locations=coords, color="blue", weight=5, opacity=0.7).add_to(m)
        AntPath(locations=coords, color='blue', delay=1000).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("Cannot render map. Route coordinates not available.")

    st.subheader("Route Breakdown Table")
    if breakdowns:
        df = pd.DataFrame(breakdowns)
        df["segment"] = df["from"].astype(str) + "→" + df["to"].astype(str)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download CSV ({algo_selected})", csv, f"{algo_selected}_breakdown.csv", "text/csv")
    else:
        st.info("No route breakdown data available.")

    st.subheader("Cost Contribution Chart")
    if breakdowns:
        expected_cols = [
            "from", "to",
            "base_time_sec", "traffic_penalty_sec",
            "weather_penalty_sec", "time_window_penalty_sec"
        ]
        available_cols = [col for col in expected_cols if col in df.columns]

        if {"from", "to"}.issubset(df.columns):
            df["segment"] = df["from"].astype(str) + "→" + df["to"].astype(str)
            index_cols = ["segment"]
            data_cols = [col for col in available_cols if col not in ["from", "to"]]

            if data_cols:
                fig, ax = plt.subplots(figsize=(12, 5))
                df_plot = df[index_cols + data_cols].copy()
                df_plot.rename(columns={c: c.replace("_sec", "").replace("_", " ").title() for c in data_cols},
                               inplace=True)
                df_plot.set_index("segment").plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
                ax.set_ylabel("Time (seconds)")
                ax.set_xlabel("Route Segment")
                ax.set_title("Segment-wise Time Contribution")
                ax.legend(loc="upper right")
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=25))
                ax.tick_params(axis='x', labelrotation=45)
                st.pyplot(fig)
            else:
                st.info("No time breakdown data available.")
        else:
            st.warning("Missing 'from'/'to' columns for plotting.")

        # 📊 Segment-wise Distance and Cost Charts
        if "segment" not in df.columns and {"from", "to"}.issubset(df.columns):
            df["segment"] = df["from"].astype(str) + "→" + df["to"].astype(str)

        # ⏱️ Segment-wise Base Time Chart
        if "segment" in df.columns and "base_time_sec" in df.columns:
            st.subheader("⏱️ Segment-wise Base Travel Time")
            fig_time, ax_time = plt.subplots(figsize=(10, 3))
            ax_time.bar(df["segment"], df["base_time_sec"], color="steelblue")
            ax_time.set_ylabel("Base Time (seconds)")
            ax_time.set_title("Base Travel Time per Segment")
            ax_time.tick_params(axis="x", rotation=45)
            st.pyplot(fig_time)

        if "distance_km" in df.columns:
            st.subheader("📍 Segment-wise Distance")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 3))
            ax_dist.bar(df["segment"], df["distance_km"], color="seagreen")
            ax_dist.set_ylabel("Distance (km)")
            ax_dist.set_title("Distance per Segment")
            ax_dist.tick_params(axis="x", rotation=45)
            st.pyplot(fig_dist)

        if "cost_dollars" in df.columns:
            st.subheader("💵 Segment-wise Cost")
            fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
            ax_cost.bar(df["segment"], df["cost_dollars"], color="tomato")
            ax_cost.set_ylabel("Cost ($)")
            ax_cost.set_title("Cost per Segment")
            ax_cost.tick_params(axis="x", rotation=45)
            st.pyplot(fig_cost)
    else:
        st.info("No data available for plotting.")

    if fitness_over_time:
        st.subheader("📈 Fitness Over Generations")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(fitness_over_time, marker="o", color="purple")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness Score")
        ax2.set_title(f"{algo_selected} Fitness Convergence")
        st.pyplot(fig2)
    else:
        st.info("No fitness data available for this algorithm.")


    if result.get("metrics_over_time"):
        st.subheader("📊 Multi-Metric Fitness Over Generations")
        df_metrics = pd.DataFrame(result["metrics_over_time"])
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(df_metrics["total_time"], label="Total Time (s)")
        ax4.plot(df_metrics["total_distance_km"], label="Total Distance (km)")
        ax4.plot(df_metrics["total_cost"], label="Total Cost ($)")
        ax4.set_xlabel("Generation")
        ax4.set_title("Multi-Metric Fitness Convergence")
        ax4.legend()
        st.pyplot(fig4)

    if result.get("pheromone_history"):
        st.subheader("🐜 ACO Pheromone Convergence")
        total_pheromone = [sum(snapshot.values()) for snapshot in result["pheromone_history"]]
        fig5, ax5 = plt.subplots(figsize=(10, 3))
        ax5.plot(total_pheromone, marker="o", color="orange")
        ax5.set_title("Total Pheromone Intensity Over Iterations")
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("Total Pheromone")
        st.pyplot(fig5)

