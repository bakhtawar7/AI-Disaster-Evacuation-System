# 🚨 Disaster Evacuation Routing System

AI-powered disaster evacuation pathfinding with 7 algorithms and an interactive React frontend.

---

## 📁 Project Structure

```
evacuation/
├── backend/
│   ├── algorithms.py      ← All 7 algorithms (pure Python)
│   ├── server.py          ← FastAPI REST API
│   └── requirements.txt
└── frontend/
    ├── public/index.html
    ├── src/
    │   ├── App.jsx        ← Full React UI
    │   └── index.js
    └── package.json
```

---

## 🛠 Setup & Run

### 1. Backend (Python + FastAPI)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn server:app --reload
# → Running at http://localhost:8000
# → Docs at  http://localhost:8000/docs
```

### 2. Frontend (React)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm start
# → Opens at http://localhost:3000
```

---

## 🧠 Algorithms Implemented

| Algorithm        | File location     | Use Case                              |
|------------------|-------------------|---------------------------------------|
| A*               | algorithms.py     | Optimal shortest path (primary)       |
| IDA*             | algorithms.py     | Memory-efficient A* for large maps    |
| BFS              | algorithms.py     | Fewest hops, unweighted paths         |
| DFS              | algorithms.py     | Explore all routes, detect dead ends  |
| Best-First Search| algorithms.py     | Fast approximate routing              |
| Minimax          | algorithms.py     | Dynamic hazard avoidance              |
| Genetic Algorithm| algorithms.py     | Optimal shelter placement             |

---

## 🗺 How to Use the UI

### Pathfinding Tab
1. Select an **algorithm** from the panel
2. Choose draw mode → click/drag on the grid:
   - **▶ Start** — evacuation start point (green)
   - **⚑ Goal** — safe zone / exit (blue)
   - **■ Wall** — buildings/blocked roads
   - **🔥 Hazard** — fire or flood zones (5× cost)
   - **🚗 Congested** — slow roads (2× cost)
3. Click **▶ RUN ALGORITHM**
4. Yellow cells = optimal path · Dark green = explored nodes

### Shelter Optimization Tab
1. Switch to **Shelter Opt.** tab
2. Draw **👤 Population** points across the grid
3. Set number of shelters and generations
4. Click **🧬 RUN GENETIC ALGO**
5. Purple cells = optimally placed shelters

---

## 🔌 API Endpoints

| Method | Endpoint             | Description                     |
|--------|----------------------|---------------------------------|
| POST   | `/find-path`         | Run any pathfinding algorithm   |
| POST   | `/optimize-shelters` | Genetic algorithm shelter opt.  |
| POST   | `/generate-grid`     | Random map generation           |
| GET    | `/docs`              | Interactive Swagger UI          |

### Example `/find-path` Request
```json
{
  "grid": [[0,0,1,...], ...],
  "start": [0, 0],
  "goal": [19, 19],
  "algorithm": "astar",
  "heuristic": "euclidean"
}
```

### Grid Cell Values
| Value | Meaning        |
|-------|----------------|
| 0     | Free cell      |
| 1     | Wall/Building  |
| 3     | Fire/Flood (5× cost) |
| 4     | Congested road (2× cost) |

---

## 🧬 Genetic Algorithm Details

Optimizes shelter placement using:
- **Fitness function**: minimize average distance from all population to nearest shelter
- **Selection**: elitism (top 25% survive)
- **Crossover**: single-point crossover on shelter coordinates
- **Mutation**: random shelter relocation (15% rate)
- **Population**: 30 individuals per generation

---

## 📌 Notes
- Minimax algorithm models evacuee (MAX) vs spreading disaster (MIN)
- IDA* is preferred over A* for very large grids (lower memory usage)
- Hazard cells are passable but expensive — algorithm routes around them when possible
