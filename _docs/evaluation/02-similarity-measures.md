# Similarity Measures

**Worum geht's?** Verschiedene Methoden um die Ähnlichkeit zwischen Vektoren zu messen.

**Warum wichtig?** Die Wahl der Similarity-Metrik beeinflusst welche Chunks als "ähnlich" erkannt werden.

---

## 1. Cosine Similarity ⭐

**Definition:** Misst den Winkel zwischen zwei Vektoren (0° = identisch, 90° = orthogonal)

**Formel:**
```
cos(θ) = (a · b) / (|a| * |b|)
Range: [-1, 1], meist [0, 1] bei Embeddings
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([query_emb], embeddings)[0]

# sentence-transformers
from sentence_transformers import util
similarity = util.cos_sim(query_emb, embeddings)[0]

# numpy (wenn normalisiert)
similarity = embeddings @ query_emb  # Dot product = Cosine!
```

**Interpretation:**
- `> 0.8`: Sehr ähnlich
- `0.6 - 0.8`: Ähnlich
- `0.4 - 0.6`: Schwach ähnlich
- `< 0.4`: Unähnlich

**Wann nutzen?**
- ✅ Standard für Semantic Search
- ✅ Unabhängig von Vektor-Länge
- ✅ Development/Testing (explizit, klar)
- ❌ Etwas langsamer als Dot Product

**Target:** Relevante Chunks sollten > 0.6 Similarity haben

---

## 2. Dot Product

**Definition:** Summe der elementweisen Multiplikation zweier Vektoren

**Formel:**
```
a · b = Σ(a[i] * b[i])
Range: [-∞, +∞]
Wenn normalisiert: identisch zu Cosine Similarity
```

**Code:**
```python
# numpy
import numpy as np

# Empfohlen: @ operator
scores = embeddings @ query_emb

# Alternative: np.dot
scores = np.dot(embeddings, query_emb)
```

**Interpretation:**
- Höherer Score = ähnlicher
- Nur zuverlässig wenn Embeddings normalisiert sind
- Dann: identisch zu Cosine (aber schneller!)

**Wann nutzen?**
- ✅ **Production RAG** (schnellste Methode)
- ✅ Wenn Embeddings normalisiert sind
- ✅ Große Datenmengen (Millionen Vektoren)
- ⚠️ Nur mit Normalisierung zuverlässig

**Target:** Normalisierung aktivieren: `model.encode(..., normalize_embeddings=True)`

---

## 3. Euclidean Distance (L2)

**Definition:** Geradlinige Distanz zwischen zwei Punkten im Vektorraum

**Formel:**
```
dist = √(Σ(a[i] - b[i])²)
Range: [0, ∞], 0 = identisch
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import euclidean_distances
distances = euclidean_distances([query_emb], embeddings)[0]

# numpy
distances = np.linalg.norm(embeddings - query_emb, axis=1)

# Zu Similarity konvertieren
similarities = 1 / (1 + distances)
```

**Interpretation:**
- Kleinere Distance = ähnlicher
- Misst absolute Position im Raum
- Geometrisch intuitiv (Luftlinie)

**Wann nutzen?**
- ✅ Clustering (K-Means nutzt Euclidean)
- ✅ Visualisierung (geometrisch intuitiv)
- ✅ Wenn Magnitude wichtig ist
- ❌ Nicht Standard für Semantic Search

**Target:** Nur für spezielle Use Cases, nicht für reguläres Retrieval

---

## 4. Manhattan Distance (L1)

**Definition:** Summe der absoluten Differenzen (Taxicab-Metrik)

**Formel:**
```
dist = Σ|a[i] - b[i]|
Range: [0, ∞]
```

**Code:**
```python
# sklearn
from sklearn.metrics.pairwise import manhattan_distances
distances = manhattan_distances([query_emb], embeddings)[0]

# numpy
distances = np.sum(np.abs(embeddings - query_emb), axis=1)
```

**Interpretation:**
- "Stadtstrecke" statt Luftlinie
- Robust gegen Outlier-Dimensionen
- Kleinere Distance = ähnlicher

**Wann nutzen?**
- ✅ Sparse, hochdimensionale Vektoren
- ✅ Robustness gegen Outliers wichtig
- ❌ Sehr selten in RAG verwendet

**Target:** Nische - nur für spezifische Anforderungen

---

## 📊 Vergleich: Wann welche Similarity?

| Methode | Speed | Normalisierung nötig? | Use Case |
|---------|-------|----------------------|----------|
| **Dot Product** | ⚡⚡⚡ Fastest | Ja (sonst unzuverlässig) | **Production RAG** (Standard) |
| **Cosine Similarity** | ⚡⚡ Fast | Nein | Development/Testing |
| **Euclidean** | ⚡ Slower | Nein | Clustering, Visualisierung |
| **Manhattan** | ⚡ Slower | Nein | Sparse Data, Robustness |

**Empfehlung für RAG:**
```python
# In Production:
scores = embeddings @ query_emb  # Dot product (schnell)

# Voraussetzung:
model.encode(..., normalize_embeddings=True)
```
