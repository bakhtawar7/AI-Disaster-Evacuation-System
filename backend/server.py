from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import random

from algorithms import (
    astar, idastar, bfs, dfs,
    best_first_search, minimax_evacuation,
    genetic_algorithm_shelters
)
from ml_models import (
    KNNClassifier, GaussianNaiveBayes,
    KMeansClustering, NeuralNetworkRegressor,
    train_all_models
)

app = FastAPI(title="Disaster Evacuation Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pre-train all ML models at startup ───────────────────────────────────────
print("Training ML models…")
ML_MODELS = train_all_models()
print(f"  KNN accuracy:        {ML_MODELS['knn']['accuracy']}%")
print(f"  Naive Bayes accuracy:{ML_MODELS['naive_bayes']['accuracy']}%")
print(f"  Neural Net R²:       {ML_MODELS['neural_network']['r2']}")
print("Models ready.")


# ─── Request Models ───────────────────────────────────────────────────────────

class PathRequest(BaseModel):
    grid: List[List[int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    algorithm: str = "astar"
    heuristic: str = "euclidean"
    hazard_sources: Optional[List[Tuple[int,int]]] = []

class ShelterRequest(BaseModel):
    grid: List[List[int]]
    population: List[Tuple[int, int]]
    num_shelters: int = 3
    generations: int = 50

class MLAnalysisRequest(BaseModel):
    grid: List[List[int]]
    path: List[Tuple[int, int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    population: Optional[List[Tuple[int, int]]] = []
    shelters: Optional[List[Tuple[int, int]]] = []
    num_clusters: int = 3


# ─── Pathfinding Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Evacuation Routing API running"}


@app.post("/find-path")
def find_path(req: PathRequest):
    algo = req.algorithm.lower()
    grid = req.grid
    start = tuple(req.start)
    goal  = tuple(req.goal)

    if algo == "astar":
        path, visited, cost = astar(grid, start, goal, req.heuristic)
    elif algo == "idastar":
        path, visited, cost = idastar(grid, start, goal, req.heuristic)
    elif algo == "bfs":
        path, visited, cost = bfs(grid, start, goal)
    elif algo == "dfs":
        path, visited, cost = dfs(grid, start, goal)
    elif algo == "bestfirst":
        path, visited, cost = best_first_search(grid, start, goal)
    elif algo == "minimax":
        hazards = [tuple(h) for h in (req.hazard_sources or [])]
        if not hazards:
            r, c = start
            hazards = [(r, c+2)] if c+2 < len(grid[0]) else [(r, c-1)]
        path, visited, cost = minimax_evacuation(grid, start, goal, hazards)
    else:
        return {"error": f"Unknown algorithm: {algo}"}

    return {
        "algorithm": algo,
        "path": path,
        "visited": visited[:300],
        "cost": round(cost, 2) if cost != float('inf') else None,
        "path_length": len(path),
        "nodes_explored": len(visited),
    }


@app.post("/optimize-shelters")
def optimize_shelters(req: ShelterRequest):
    best_shelters, history = genetic_algorithm_shelters(
        req.grid,
        [tuple(p) for p in req.population],
        req.num_shelters,
        req.generations,
    )
    return {
        "shelters": best_shelters,
        "generations": len(history),
        "final_fitness": history[-1]["fitness"] if history else 0,
        "history": history[-10:],
    }


@app.post("/generate-grid")
def generate_grid(rows: int = 20, cols: int = 20, hazard_pct: float = 0.1, wall_pct: float = 0.15):
    grid = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            rnd = random.random()
            if rnd < wall_pct:
                grid[r][c] = 1
            elif rnd < wall_pct + hazard_pct:
                grid[r][c] = 3
            elif rnd < wall_pct + hazard_pct + 0.05:
                grid[r][c] = 4
    grid[0][0] = 0
    grid[rows-1][cols-1] = 0
    grid[0][cols-1] = 0
    grid[rows-1][0] = 0
    return {"grid": grid, "rows": rows, "cols": cols}


# ─── ML Analysis Route ────────────────────────────────────────────────────────

@app.post("/ml-analysis")
def ml_analysis(req: MLAnalysisRequest):
    grid       = req.grid
    path       = [tuple(p) for p in req.path]
    start      = tuple(req.start)
    goal       = tuple(req.goal)
    population = [tuple(p) for p in req.population]
    shelters   = [tuple(s) for s in req.shelters]

    # ── 1. KNN: Classify each cell's danger level ──────────────────────────
    knn_model = ML_MODELS["knn"]["model"]
    features, coords = KNNClassifier.extract_features(grid, goal)
    danger_labels = knn_model.predict(features)
    danger_map = {coords[i]: danger_labels[i] for i in range(len(coords))}

    # Danger stats
    label_counts = {0: 0, 1: 0, 2: 0}
    for lbl in danger_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Danger along the path
    path_danger = [danger_map.get(tuple(p), 0) for p in path]
    avg_path_danger = round(sum(path_danger) / max(len(path_danger), 1), 2)

    # Sample a few representative danger cells for the frontend heatmap
    danger_cells = [
        {"pos": list(coords[i]), "level": danger_labels[i]}
        for i in range(len(coords))
        if danger_labels[i] == 2
    ][:80]   # cap for response size

    # ── 2. K-Means: Cluster population into evacuation zones ───────────────
    kmeans_result = None
    if population and len(population) >= req.num_clusters:
        km = KMeansClustering(k=req.num_clusters)
        km.fit(population)
        assignments = km.assign_to_shelters(shelters) if shelters else {}
        silhouette = km.silhouette_score_approx(population)
        kmeans_result = {
            "centroids": [c.tolist() for c in km.centroids],
            "labels": km.labels_.tolist(),
            "assignments": {str(k): v for k, v in assignments.items()},
            "silhouette_score": silhouette,
            "inertia": round(km.inertia(population), 2),
        }

    # ── 3. Naive Bayes: Route safety prediction ────────────────────────────
    nb_model   = ML_MODELS["naive_bayes"]["model"]
    nb_result  = None
    route_features = GaussianNaiveBayes.extract_route_features(path, grid, start, goal)
    if route_features:
        prediction = nb_model.predict_one(route_features)
        proba      = nb_model.predict_proba(route_features)
        nb_result  = {
            "prediction": int(prediction),
            "label": "RISKY ROUTE" if prediction == 1 else "SAFE ROUTE",
            "probability_safe": proba.get(0, 0),
            "probability_risky": proba.get(1, 0),
            "features": {
                "path_cost":        round(route_features[0], 2),
                "hazard_cells":     int(route_features[1]),
                "congested_cells":  int(route_features[2]),
                "path_length":      int(route_features[3]),
                "detour_ratio":     round(route_features[4], 2),
            }
        }

    # ── 4. Neural Network: Predict evacuation time ─────────────────────────
    nn_model  = ML_MODELS["neural_network"]["model"]
    nn_result = None
    nn_features = NeuralNetworkRegressor.extract_features(path, grid, population, shelters)
    if path:
        predicted_time = nn_model.predict(nn_features)
        nn_result = {
            "estimated_minutes": round(max(1.0, predicted_time), 1),
            "features": {
                "path_length":      len(path),
                "hazard_cells":     nn_features[0][1],
                "congested_cells":  nn_features[0][2],
                "population_density": round(nn_features[0][3], 3),
                "num_shelters":     len(shelters),
            }
        }

    # ── Model performance metrics (from training) ──────────────────────────
    metrics = {
        "knn": {
            "accuracy": ML_MODELS["knn"]["accuracy"],
            "confusion_matrix": ML_MODELS["knn"]["confusion_matrix"],
            "classes": ML_MODELS["knn"]["classes"],
            "n_train": ML_MODELS["knn"]["n_train"],
        },
        "naive_bayes": {
            "accuracy": ML_MODELS["naive_bayes"]["accuracy"],
            "confusion_matrix": ML_MODELS["naive_bayes"]["confusion_matrix"],
            "classes": ML_MODELS["naive_bayes"]["classes"],
            "n_train": ML_MODELS["naive_bayes"]["n_train"],
        },
        "neural_network": {
            "r2":   ML_MODELS["neural_network"]["r2"],
            "rmse": ML_MODELS["neural_network"]["rmse"],
            "mae":  ML_MODELS["neural_network"]["mae"],
            "loss_history": ML_MODELS["neural_network"]["loss_history"],
            "n_train": ML_MODELS["neural_network"]["n_train"],
        },
    }

    return {
        "knn": {
            "danger_map_sample": danger_cells,
            "zone_counts": {
                "safe":     label_counts[0],
                "moderate": label_counts[1],
                "danger":   label_counts[2],
            },
            "avg_path_danger": avg_path_danger,
            "path_danger_labels": path_danger,
        },
        "kmeans": kmeans_result,
        "naive_bayes": nb_result,
        "neural_network": nn_result,
        "metrics": metrics,
    }


@app.get("/model-metrics")
def model_metrics():
    """Return pre-computed training metrics for all models."""
    return {
        "knn": {
            "accuracy": ML_MODELS["knn"]["accuracy"],
            "confusion_matrix": ML_MODELS["knn"]["confusion_matrix"],
            "classes": ML_MODELS["knn"]["classes"],
            "k": 5,
            "n_train": ML_MODELS["knn"]["n_train"],
            "n_test":  ML_MODELS["knn"]["n_test"],
        },
        "naive_bayes": {
            "accuracy": ML_MODELS["naive_bayes"]["accuracy"],
            "confusion_matrix": ML_MODELS["naive_bayes"]["confusion_matrix"],
            "classes": ML_MODELS["naive_bayes"]["classes"],
            "n_train": ML_MODELS["naive_bayes"]["n_train"],
            "n_test":  ML_MODELS["naive_bayes"]["n_test"],
        },
        "neural_network": {
            "architecture": "6 → 16 → 8 → 1",
            "r2":   ML_MODELS["neural_network"]["r2"],
            "rmse": ML_MODELS["neural_network"]["rmse"],
            "mae":  ML_MODELS["neural_network"]["mae"],
            "loss_history": ML_MODELS["neural_network"]["loss_history"],
            "n_train": ML_MODELS["neural_network"]["n_train"],
            "n_test":  ML_MODELS["neural_network"]["n_test"],
        },
    }
