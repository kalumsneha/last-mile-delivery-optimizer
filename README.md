# 🚚 Last-Mile Delivery Route Optimization with GA, ACO, and Hybrid Metaheuristics

This project tackles the **Last-Mile Delivery Optimization Problem with Time Windows (VRPTW)** using three powerful metaheuristics:
- 🔬 **Genetic Algorithm (GA)**
- 🐜 **Ant Colony Optimization (ACO)**
- 🧬 **Hybrid GA + ACO Approach**

Unlike traditional models that assume straight-line (Euclidean) distances, this project leverages real-world **road network data** via OSRM and integrates **dynamic cost modeling** to simulate real delivery conditions.

---

## 🎥 Demo

📎 [View Demo Videos and Plots](https://drive.google.com/drive/folders/11TQrXs2EIc7Y5X8z0tLamWIgNhJUQDlc?usp=sharing)

---

## 🧠 Features

- ✅ Real-world routing with **OSRM** (Open Source Routing Machine)
- ⏰ Time window and penalty-aware **cost modeling**
- 📊 Interactive **Streamlit dashboard** with:
  - Route maps
  - Ant-style animations
  - Generation-wise convergence plots
- 🧪 Benchmarking with **Solomon VRPTW datasets**
- 📈 Built-in evaluation and visual reports (CSV + PNG)

---

## 🧬 Algorithms

| Algorithm  | Description |
|------------|-------------|
| **GA**     | Evolutionary algorithm using PMX crossover, mutation, and selection. |
| **ACO**    | Bio-inspired model simulating ant foraging behavior with pheromone trails and heuristics. |
| **Hybrid** | Combines GA's global search with ACO's local refinement of top individuals. |

---

## 📁 Project Structure

```bash
.
last-mile-delivery-optimizer/
├── backend/               # Core optimizers, API, analysis, and benchmark tools
├── frontend/              # Streamlit frontend for UI and visualization
├── data/                  # JSON data for delivery points and test cases
├── benchmark_results/     # CSVs, plots, and logs from optimization runs
├── reports/plots/         # Rendered result plots used in paper/demo
├── util/                  # Supporting utilities (e.g., coordinate converters)
├── requirements.txt       # Python dependencies
├── main.py                # Entry point (if needed for CLI testing)
├── README.md              # This file


```
1. Clone the Repository
   
```bash
git clone https://github.com/kalumsneha/last-mile-delivery-optimizer.git
cd last-mile-delivery-optimizer
```
2.  Install Dependencies
   
```bash
pip install -r requirements.txt
```
Use venv or conda to manage a clean environment.

3. Set Up OSRM (Routing Backend)
```bash
# Download regional OSM map (e.g., Ontario)
wget http://download.geofabrik.de/north-america/canada/ontario-latest.osm.pbf

# Preprocess the data
osrm-extract -p profiles/car.lua ontario-latest.osm.pbf
osrm-contract ontario-latest.osrm

# Start the OSRM routing server
osrm-routed ontario-latest.osrm
```
4.  Run the Application

Start the Backend
```bash
uvicorn main:app --reload
```
Start the Frontend
```bash
streamlit run app.p
```
Open http://localhost:8501 in your browser.

5. Evaluation & Benchmarking
```bash
Run benchmark experiments with plotting:
python run_full_benchmark.py
```



