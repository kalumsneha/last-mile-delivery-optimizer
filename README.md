


# 🚚 Last-Mile Delivery Route Optimization with GA, ACO, and Hybrid Metaheuristics

This project addresses the **Last-Mile Delivery Optimization Problem with Time Windows** using advanced metaheuristic algorithms: **Genetic Algorithm (GA)**, **Ant Colony Optimization (ACO)**, and a **Hybrid GA+ACO approach**.

It features:
- Real-world **road network routing** via OSRM (not just Euclidean)
- Dynamic cost modeling (traffic, weather, time-window penalties)
- A full-stack system with:
  - 🧠 Python backend (FastAPI)
  - 📊 Streamlit frontend for visualization
  - 🗺️ Route animation and metrics dashboard
- Benchmarking support with Solomon VRPTW datasets
- Built-in evaluation and convergence plots

---

## Algorithms

| Algorithm | Description |
|----------|-------------|
| **GA**    | Uses evolutionary operators (PMX crossover, mutation) to evolve solutions. |
| **ACO**   | Simulates ant foraging behavior with pheromone trails and heuristic desirability. |
| **Hybrid**| Combines GA exploration with periodic ACO local refinement of top solutions. |

---

## 📂 Project Structure

```bash
.
├── backend/
│   ├── main.py                  # FastAPI backend
│   ├── optimizer.py             # GA implementation
│   ├── aco_optimizer.py         # ACO implementation
│   ├── hybrid_optimizer.py      # Hybrid GA + ACO
│   ├── cost.py                  # OSRM-based cost calculation
│   ├── evaluate.py              # Route fitness + penalty logic
│   ├── solomon_parser.py        # Parser for Solomon datasets
│   ├── utils.py                 # Misc utilities
│   └── analysis_plots.py        # Benchmark runner and plotting
│
├── app.py                       # Streamlit frontend
├── benchmark_results/           # Stores output CSVs and plots
├── datasets/                    # Solomon VRPTW sample files
├── img/                         # Figures used in paper and README
└── README.md                    # You are here!


1. Clone the Repository
git clone https://github.com/<your-username>/last-mile-delivery-optimizer.git
cd last-mile-delivery-optimizer




2. Install Dependencies
Use conda or venv to manage the environment.
```bash
pip install -r requirements.txt

Key packages:
fastapi, uvicorn


streamlit, folium, matplotlib


requests, deap, numpy, pandas


3. Set Up OSRM
Install and Run OSRM Backend:

```bash
# Download OSM extract (e.g., Ontario)
wget http://download.geofabrik.de/north-america/canada/ontario-latest.osm.pbf

# Build the car routing profile
osrm-extract -p profiles/car.lua ontario-latest.osm.pbf
osrm-contract ontario-latest.osrm
osrm-routed ontario-latest.osrm


Ensure OSRM is running at: http://localhost:5000/route/v1/driving/...
Run the Application

Start the Backend
```bash
uvicorn backend.main:app --reload --port 8000


Start the Frontend

```bash
streamlit run app.py

Then open: http://localhost:8501

Evaluation & Benchmarking

You can run built-in benchmarking and plotting via:
```bash
python backend/analysis_plots.py


License
This project is licensed under the MIT License – see the LICENSE file for details.

