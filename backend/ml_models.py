"""
ml_models.py — Machine Learning layer for Disaster Evacuation System
=====================================================================
Dataset: Forest Fire Dataset (UCI Machine Learning Repository)
         Cortez & Morais, 2007 — forest_fire_dataset.csv
         Features: FFMC, DMC, DC, ISI, temp, RH, wind, rain, area

Models:
  1. KNN            → Classify zone danger level    (SAFE/MODERATE/DANGER)
  2. K-Means        → Cluster population into zones
  3. Naive Bayes    → Predict route safety          (SAFE/RISKY)
  4. Neural Network → Predict evacuation time       (minutes)

All models trained on REAL forest fire data loaded from CSV.
"""

import os
import csv
import math
import random
import numpy as np
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(__file__), "forest_fire_dataset.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Dataset Loader
# ══════════════════════════════════════════════════════════════════════════════

MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
DAY_MAP   = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5,"sat":6,"sun":7}

def load_dataset():
    """
    Load forest_fire_dataset.csv and return feature matrix X and labels.
    Columns used: FFMC, DMC, DC, ISI, temp, RH, wind, rain
    Labels derived from: danger_zone (0/1/2), route_risk (0/1), evac_time
    """
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ffmc  = float(row["FFMC"])
                dmc   = float(row["DMC"])
                dc    = float(row["DC"])
                isi   = float(row["ISI"])
                temp  = float(row["temp"])
                rh    = float(row["RH"])
                wind  = float(row["wind"])
                rain  = float(row["rain"])
                area  = float(row["area"])
                dzone = int(row["danger_zone"])
                rrisk = int(row["route_risk"])
                etime = float(row["evac_time"])
                rows.append({
                    "features": [ffmc, dmc, dc, isi, temp, rh, wind, rain],
                    "danger_zone": dzone,
                    "route_risk":  rrisk,
                    "evac_time":   etime,
                    "area":        area,
                })
            except (ValueError, KeyError):
                continue
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Shared Utilities
# ══════════════════════════════════════════════════════════════════════════════

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def normalize(X):
    X = np.array(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1
    return (X - mins) / denom, mins, maxs

def apply_norm(X, mins, maxs):
    X = np.array(X, dtype=float)
    denom = (maxs - mins)
    denom[denom == 0] = 1
    return (X - mins) / denom

def train_test_split(X, y, test_size=0.2, seed=42):
    random.seed(seed)
    idx = list(range(len(X)))
    random.shuffle(idx)
    split = int(len(idx) * (1 - test_size))
    tr, te = idx[:split], idx[split:]
    return [X[i] for i in tr],[X[i] for i in te],[y[i] for i in tr],[y[i] for i in te]

def accuracy_score(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return round(correct / len(y_true) * 100, 2)

def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0]*n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm

def rmse(y_true, y_pred):
    return round(math.sqrt(sum((a-b)**2 for a,b in zip(y_true,y_pred))/len(y_true)), 3)

def mae(y_true, y_pred):
    return round(sum(abs(a-b) for a,b in zip(y_true,y_pred))/len(y_true), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 1. KNN — Zone Danger Classification
# ══════════════════════════════════════════════════════════════════════════════

class KNNClassifier:
    """
    K-Nearest Neighbours — classifies each grid zone as SAFE/MODERATE/DANGER
    Trained on real forest fire features: [FFMC, DMC, DC, ISI, temp, RH, wind, rain]
    Label: danger_zone (0=SAFE, 1=MODERATE, 2=DANGER)
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.mins = self.maxs = None

    def fit(self, X, y):
        X_norm, self.mins, self.maxs = normalize(X)
        self.X_train = X_norm
        self.y_train = np.array(y)

    def predict_one(self, x):
        x_norm = apply_norm([x], self.mins, self.maxs)[0]
        dists  = np.linalg.norm(self.X_train - x_norm, axis=1)
        k_idx  = np.argsort(dists)[:self.k]
        return int(Counter(self.y_train[k_idx]).most_common(1)[0][0])

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba(self, x):
        x_norm = apply_norm([x], self.mins, self.maxs)[0]
        dists  = np.linalg.norm(self.X_train - x_norm, axis=1)
        k_idx  = np.argsort(dists)[:self.k]
        counts = Counter(self.y_train[k_idx])
        return {cls: round(counts.get(cls,0)/self.k, 2) for cls in [0,1,2]}

    # ── Grid feature extraction ───────────────────────────────────────────────
    @staticmethod
    def extract_features(grid, goal):
        """Map grid cell properties → [FFMC-like, DMC-like, DC-like, ISI-like]"""
        rows, cols = len(grid), len(grid[0])
        features, coords = [], []
        for r in range(rows):
            for c in range(cols):
                # Hazard proximity → maps to FFMC (fire spread index)
                haz = 0.0
                for dr in range(-3,4):
                    for dc in range(-3,4):
                        nr,nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==3:
                            haz += 1.0/max(abs(dr)+abs(dc),1)
                ffmc_like = min(96.0, 80.0 + haz*3)
                # Congestion → maps to DMC
                dmc_like  = 100.0 if grid[r][c]==4 else 30.0
                # Wall density → maps to DC
                walls     = sum(1 for dr in [-1,0,1] for dc2 in [-1,0,1]
                                if 0<=r+dr<rows and 0<=c+dc2<cols
                                and grid[r+dr][c+dc2]==1)
                dc_like   = walls * 100.0
                # Distance to exit → maps to ISI
                dist      = euclidean((r,c), goal)
                isi_like  = max(0, 20.0 - dist)
                # Temp (hazard cell = hot)
                temp_like = 35.0 if grid[r][c]==3 else 18.0
                rh_like   = 20.0 if grid[r][c]==3 else 55.0
                wind_like = 6.0  if grid[r][c]==4 else 3.0
                rain_like = 0.0
                features.append([ffmc_like,dmc_like,dc_like,isi_like,
                                  temp_like,rh_like,wind_like,rain_like])
                coords.append((r,c))
        return features, coords

    @staticmethod
    def load_training_data():
        data = load_dataset()
        X = [d["features"] for d in data]
        y = [d["danger_zone"] for d in data]
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. K-Means Clustering — Population Evacuation Zones
# ══════════════════════════════════════════════════════════════════════════════

class KMeansClustering:
    """
    K-Means — groups population into evacuation zones.
    Uses K-Means++ initialisation for stable clusters.
    """
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        if len(X) < self.k:
            self.k = max(1, len(X))
        random.seed(0)
        centroids = [X[random.randint(0, len(X)-1)]]
        for _ in range(self.k-1):
            dists = np.array([min(np.linalg.norm(x-c)**2 for c in centroids) for x in X])
            probs = dists / dists.sum()
            r = random.random()
            for j, p in enumerate(np.cumsum(probs)):
                if r <= p:
                    centroids.append(X[j]); break
        self.centroids = np.array(centroids)
        for _ in range(self.max_iter):
            dists  = np.linalg.norm(X[:,None]-self.centroids[None,:], axis=2)
            labels = np.argmin(dists, axis=1)
            new_c  = np.array([X[labels==i].mean(axis=0) if (labels==i).sum()>0
                               else self.centroids[i] for i in range(self.k)])
            if np.allclose(new_c, self.centroids, atol=1e-4): break
            self.centroids = new_c
        self.labels_ = labels
        return self

    def assign_to_shelters(self, shelters):
        if not shelters: return {}
        assignments = {}
        for i, centroid in enumerate(self.centroids):
            nearest = min(shelters, key=lambda s: euclidean(centroid, s))
            assignments[i] = {
                "centroid": centroid.tolist(),
                "shelter":  list(nearest),
                "distance": round(euclidean(centroid, nearest), 2)
            }
        return assignments

    def inertia(self, X):
        X = np.array(X, dtype=float)
        dists = np.linalg.norm(X[:,None]-self.centroids[None,:], axis=2)
        return float(np.min(dists, axis=1).sum())

    def silhouette_score_approx(self, X):
        X = np.array(X, dtype=float)
        if len(X)<2 or self.k<2: return 0.0
        labels = self.labels_
        scores = []
        for i, x in enumerate(X):
            same = X[labels==labels[i]]
            a = np.linalg.norm(same-x, axis=1).mean() if len(same)>1 else 0
            other = [np.linalg.norm(X[labels==j]-x,axis=1).mean()
                     for j in range(self.k) if j!=labels[i] and (labels==j).sum()>0]
            b = min(other) if other else 0
            s = (b-a)/max(a,b) if max(a,b)>0 else 0
            scores.append(s)
        return round(float(np.mean(scores)), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Gaussian Naive Bayes — Route Safety Prediction
# ══════════════════════════════════════════════════════════════════════════════

class GaussianNaiveBayes:
    """
    Naive Bayes — predicts SAFE / RISKY route.
    Trained on real fire data features mapped to route properties.
    Label: route_risk (0=SAFE, 1=RISKY)
    """
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = {}
        self.means_ = {}
        self.vars_  = {}

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            Xc = X[y==cls]
            self.class_priors_[cls] = len(Xc)/len(X)
            self.means_[cls] = Xc.mean(axis=0)
            self.vars_[cls]  = Xc.var(axis=0) + 1e-9

    def _log_likelihood(self, x, cls):
        m, v = self.means_[cls], self.vars_[cls]
        return -0.5*np.sum(np.log(2*np.pi*v)+((x-m)**2)/v) + math.log(self.class_priors_[cls])

    def predict_one(self, x):
        scores = {cls: self._log_likelihood(x, cls) for cls in self.classes_}
        return max(scores, key=scores.get)

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba(self, x):
        x = np.array(x, dtype=float)
        ls = {cls: self._log_likelihood(x, cls) for cls in self.classes_}
        mx = max(ls.values())
        es = {cls: math.exp(v-mx) for cls,v in ls.items()}
        total = sum(es.values())
        return {int(cls): round(v/total,3) for cls,v in es.items()}

    @staticmethod
    def extract_route_features(path, grid, start, goal):
        if not path or len(path)<2: return None
        hazard   = sum(1 for r,c in path if grid[r][c]==3)
        congested= sum(1 for r,c in path if grid[r][c]==4)
        length   = len(path)
        s_dist   = max(euclidean(start, goal), 1)
        detour   = length / s_dist
        cost     = sum(euclidean(path[i],path[i+1]) *
                       (5 if grid[path[i+1][0]][path[i+1][1]]==3
                        else 2 if grid[path[i+1][0]][path[i+1][1]]==4
                        else 1)
                       for i in range(len(path)-1))
        # Map to fire dataset features: [FFMC, DMC, DC, ISI, temp, RH, wind, rain]
        ffmc = min(96, 80 + hazard*2)
        dmc  = min(200, 30 + congested*20)
        dc   = min(800, cost*5)
        isi  = min(30, detour*5)
        temp = 20 + hazard*2
        rh   = max(20, 60 - hazard*5)
        wind = min(9, 2 + congested*1.5)
        rain = 0.0
        return [ffmc, dmc, dc, isi, temp, rh, wind, rain]

    @staticmethod
    def load_training_data():
        data = load_dataset()
        X = [d["features"] for d in data]
        y = [d["route_risk"] for d in data]
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 4. Neural Network — Evacuation Time Prediction
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetworkRegressor:
    """
    Neural Network (6→16→8→1) trained on real forest fire data.
    Predicts evacuation time in minutes.
    Input features: [FFMC, DMC, DC, ISI, temp, RH, wind, rain]  (8 features)
    Target: evac_time (minutes derived from fire area + conditions)
    """
    def __init__(self, hidden1=16, hidden2=8, lr=0.01, epochs=200, batch_size=32):
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.hidden1 = hidden1; self.hidden2 = hidden2
        self.weights = None; self.mins = self.maxs = None
        self.y_min = self.y_max = None; self.loss_history = []

    def _init_weights(self, n_in):
        np.random.seed(7)
        self.weights = {
            "W1": np.random.randn(n_in, self.hidden1)*np.sqrt(2.0/n_in),
            "b1": np.zeros((1, self.hidden1)),
            "W2": np.random.randn(self.hidden1, self.hidden2)*np.sqrt(2.0/self.hidden1),
            "b2": np.zeros((1, self.hidden2)),
            "W3": np.random.randn(self.hidden2, 1)*np.sqrt(2.0/self.hidden2),
            "b3": np.zeros((1, 1)),
        }

    @staticmethod
    def relu(x):      return np.maximum(0, x)
    @staticmethod
    def relu_grad(x): return (x>0).astype(float)

    def forward(self, X):
        z1=X@self.weights["W1"]+self.weights["b1"]; a1=self.relu(z1)
        z2=a1@self.weights["W2"]+self.weights["b2"]; a2=self.relu(z2)
        z3=a2@self.weights["W3"]+self.weights["b3"]
        return z3, (z1,a1,z2,a2)

    def backward(self, X, y, out, cache):
        z1,a1,z2,a2 = cache; m=len(X)
        dz3=( out-y)/m
        dW3=a2.T@dz3; db3=dz3.sum(axis=0,keepdims=True)
        dz2=(dz3@self.weights["W3"].T)*self.relu_grad(z2)
        dW2=a1.T@dz2; db2=dz2.sum(axis=0,keepdims=True)
        dz1=(dz2@self.weights["W2"].T)*self.relu_grad(z1)
        dW1=X.T@dz1;  db1=dz1.sum(axis=0,keepdims=True)
        for k,g in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2),("W3",dW3),("b3",db3)]:
            self.weights[k] -= self.lr*g

    def fit(self, X, y):
        X_norm,self.mins,self.maxs = normalize(X)
        y = np.array(y, dtype=float).reshape(-1,1)
        self.y_min,self.y_max = y.min(), y.max()
        y_norm = (y-self.y_min)/max(self.y_max-self.y_min, 1)
        self._init_weights(X_norm.shape[1])
        self.loss_history = []
        for epoch in range(self.epochs):
            idx = np.random.permutation(len(X_norm))
            Xs,ys = X_norm[idx], y_norm[idx]
            loss = 0
            for i in range(0,len(Xs),self.batch_size):
                Xb,yb = Xs[i:i+self.batch_size], ys[i:i+self.batch_size]
                out,cache = self.forward(Xb)
                loss += float(np.mean((out-yb)**2))
                self.backward(Xb,yb,out,cache)
            self.loss_history.append(round(loss,4))
        return self

    def predict(self, X):
        Xn = apply_norm(X, self.mins, self.maxs)
        out,_ = self.forward(Xn)
        return float(out[0][0])*(self.y_max-self.y_min)+self.y_min

    @staticmethod
    def extract_features(path, grid, population, shelters):
        rows,cols = len(grid),len(grid[0])
        length   = len(path) if path else 0
        hazard   = sum(1 for r,c in path if grid[r][c]==3) if path else 0
        congested= sum(1 for r,c in path if grid[r][c]==4) if path else 0
        ffmc     = min(96, 80+hazard*2)
        dmc      = min(200, 30+congested*20)
        dc       = min(800, length*10)
        isi      = min(30, length*0.5)
        temp     = 20+hazard*2
        rh       = max(20, 60-hazard*5)
        wind     = min(9, 2+congested*1.5)
        rain     = 0.0
        return [[ffmc, dmc, dc, isi, temp, rh, wind, rain]]

    @staticmethod
    def load_training_data():
        data = load_dataset()
        X = [d["features"] for d in data]
        y = [d["evac_time"] for d in data]
        return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Master training function
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models():
    results = {}
    data = load_dataset()
    print(f"  Loaded {len(data)} real fire records from CSV")

    # ── KNN ──────────────────────────────────────────────────────────────────
    X_knn = [d["features"] for d in data]
    y_knn = [d["danger_zone"] for d in data]
    Xtr,Xte,ytr,yte = train_test_split(X_knn, y_knn)
    knn = KNNClassifier(k=5)
    knn.fit(Xtr, ytr)
    preds = knn.predict(Xte)
    acc   = accuracy_score(yte, preds)
    cm    = confusion_matrix(yte, preds, [0,1,2])
    results["knn"] = {"model":knn,"accuracy":acc,"confusion_matrix":cm,
                      "classes":["SAFE","MODERATE","DANGER"],
                      "n_train":len(Xtr),"n_test":len(Xte)}

    # ── Naive Bayes ───────────────────────────────────────────────────────────
    X_nb = [d["features"] for d in data]
    y_nb = [d["route_risk"] for d in data]
    Xtr,Xte,ytr,yte = train_test_split(X_nb, y_nb)
    nb = GaussianNaiveBayes()
    nb.fit(Xtr, ytr)
    preds = nb.predict(Xte)
    acc   = accuracy_score(yte, preds)
    cm    = confusion_matrix(yte, preds, [0,1])
    results["naive_bayes"] = {"model":nb,"accuracy":acc,"confusion_matrix":cm,
                               "classes":["SAFE ROUTE","RISKY ROUTE"],
                               "n_train":len(Xtr),"n_test":len(Xte)}

    # ── Neural Network ────────────────────────────────────────────────────────
    X_nn = [d["features"] for d in data]
    y_nn = [d["evac_time"] for d in data]
    Xtr,Xte,ytr,yte = train_test_split(X_nn, y_nn)
    nn = NeuralNetworkRegressor(lr=0.01, epochs=200)
    nn.fit(Xtr, ytr)
    preds = [nn.predict([x]) for x in Xte]
    r2_num = sum((a-b)**2 for a,b in zip(yte,preds))
    r2_den = sum((a-sum(yte)/len(yte))**2 for a in yte)
    r2 = round(1-r2_num/r2_den, 3) if r2_den>0 else 0
    results["neural_network"] = {
        "model":nn,"rmse":rmse(yte,preds),"mae":mae(yte,preds),"r2":r2,
        "loss_history":nn.loss_history[::10],
        "n_train":len(Xtr),"n_test":len(Xte)
    }

    return results
