# Embedding Evaluation

**Warum wichtig?** Embeddings sind das Fundament des Retrievals. Schlechte Embeddings führen zu schlechtem Retrieval, selbst mit perfektem Ranking-Algorithmus.

---

## Semantic Similarity Tests

**Definition:** Prüfen ob semantisch ähnliche Texte ähnliche Embeddings haben

**Code:**
```python
# sentence-transformers
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('deepset/gbert-large')

# Test-Paare erstellen
test_pairs = [
    # (text1, text2, expected_similarity)
    ("Kühlschrank", "Kühlgerät", "high"),        # Synonyme
    ("Energieverbrauch", "kWh", "high"),         # Verwandt
    ("Temperatur", "Preis", "low"),              # Unrelated
    ("DIN 13277", "DIN13277", "high"),           # Format-Varianten
    ("SmartMonitoring", "Smart Monitoring", "high"),  # Schreibweisen
]

results = []
for text1, text2, expected in test_pairs:
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    similarity = util.cos_sim(emb1, emb2).item()

    results.append({
        'pair': (text1, text2),
        'similarity': similarity,
        'expected': expected,
        'pass': (expected == 'high' and similarity > 0.7) or
                (expected == 'low' and similarity < 0.5)
    })

# Analyse
pass_rate = sum(r['pass'] for r in results) / len(results)
print(f"Pass Rate: {pass_rate:.2%}")
```

**Interpretation:**
- Pass Rate > 80%: Embeddings funktionieren gut
- Pass Rate 60-80%: Akzeptabel, Verbesserungspotential
- Pass Rate < 60%: Problematisch - Model passt nicht zum Use Case

**Wann nutzen?**
- ✅ Bei Auswahl des Embedding-Models
- ✅ Nach Fine-Tuning
- ✅ Beim Testen domain-spezifischer Begriffe

**Target:** > 80% Pass Rate

---

## Clustering Quality

**Definition:** Prüfen ob ähnliche Chunks im Embedding-Space clustern

**Code:**
```python
# sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Embeddings laden
embeddings = np.load('embeddings_gbert.npy')

# K-Means Clustering
n_clusters = 10  # Erwartete Anzahl Produktgruppen
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Silhouette Score (Cluster-Qualität)
silhouette = silhouette_score(embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette:.3f}")
# Range: [-1, 1]
# > 0.5: Gute Separation
# < 0.3: Schlechte Separation
```

**Formel:**
```
Silhouette Score = (b - a) / max(a, b)

a = durchschnittliche Distanz zu Punkten im selben Cluster
b = durchschnittliche Distanz zum nächsten Cluster

Range: [-1, 1], höher = besser
```

**Interpretation:**
- Silhouette > 0.5: Embeddings bilden klare Cluster - gut!
- Silhouette 0.3-0.5: Moderate Cluster-Struktur
- Silhouette < 0.3: Embeddings sind zu gemischt - schlecht

**Wann nutzen?**
- ✅ Bei strukturierten Daten (Produktkategorien, Dokumenttypen)
- ✅ Zum Vergleich verschiedener Embedding-Models
- ✅ Nach Dimension Reduction

**Target:** Silhouette > 0.4

---

## Outlier Detection

**Definition:** Chunks finden die "alleine" im Embedding-Space sind

**Code:**
```python
# sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Finde für jeden Chunk die 5 nächsten Nachbarn
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
nbrs.fit(embeddings)
distances, indices = nbrs.kneighbors(embeddings)

# Durchschnittliche Distanz zu Nachbarn
avg_distances = distances.mean(axis=1)

# Outliers: Chunks mit hoher avg. Distanz
outlier_threshold = np.percentile(avg_distances, 95)
outliers = np.where(avg_distances > outlier_threshold)[0]

print(f"Found {len(outliers)} outliers ({len(outliers)/len(embeddings)*100:.1f}%)")

# Outliers untersuchen
for idx in outliers[:5]:
    print(f"Outlier {idx}: {chunks[idx][:100]}...")
```

**Interpretation:**
- Viele Outliers (> 10%): Datenqualitätsprobleme oder sehr heterogene Daten
- Wenige Outliers (< 5%): Normal - jedes Dataset hat ein paar Edge Cases
- Outliers prüfen: Oft korrupte Daten, Formatierungsfehler, oder sehr spezifische Inhalte

**Wann nutzen?**
- ✅ Data Quality Checks
- ✅ Beim Debugging schlechter Retrieval-Ergebnisse
- ✅ Vor Production-Deployment

**Target:** < 5% Outliers

---

## Cross-Lingual Test

**Definition:** Funktioniert Embedding bei gemischten Sprachen?

**Code:**
```python
# sentence-transformers
test_pairs = [
    ("SmartMonitoring", "Smart Monitoring", "high"),  # EN/DE Mix
    ("WiFi Schnittstelle", "WiFi interface", "high"),
    ("Temperaturüberwachung", "Temperature monitoring", "high"),
]

# Teste mit multilingual model
model = SentenceTransformer('intfloat/multilingual-e5-large')

for text1, text2, expected in test_pairs:
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    similarity = util.cos_sim(emb1, emb2).item()
    print(f"{text1} <-> {text2}: {similarity:.3f}")
```

**Interpretation:**
- Similarity > 0.7: Model versteht cross-lingual Konzepte gut
- Similarity 0.5-0.7: Moderate cross-lingual Performance
- Similarity < 0.5: Model ist nicht multilingual-fähig

**Wann nutzen?**
- ✅ Bei mehrsprachigen Daten
- ✅ Bei Fachbegriffen in verschiedenen Sprachen
- ✅ Internationale Anwendungen

**Target:** Similarity > 0.7 für äquivalente Begriffe

---

## Dimensionality Analysis

**Definition:** Wie viele Dimensionen werden wirklich genutzt?

**Code:**
```python
# sklearn
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(embeddings)

# Explained Variance
explained_var = pca.explained_variance_ratio_

# Wie viele Dimensionen für 95% Varianz?
cumsum = np.cumsum(explained_var)
n_dims_95 = np.argmax(cumsum >= 0.95) + 1

print(f"95% Varianz in {n_dims_95} von {embeddings.shape[1]} Dimensionen")
# Wenn n_dims_95 << 1024 → viele Dimensionen unnötig

# Visualisierung der Top-Komponenten
import matplotlib.pyplot as plt
plt.plot(cumsum[:100])
plt.xlabel('Anzahl Komponenten')
plt.ylabel('Kumulative erklärte Varianz')
plt.title('PCA Explained Variance')
plt.show()
```

**Formel:**
```
Explained Variance Ratio = λ_i / Σλ

λ_i = Eigenvalue der i-ten Komponente
Σλ = Summe aller Eigenvalues
```

**Interpretation:**
- n_dims_95 << embedding_dim: Viele Dimensionen redundant - Dimension Reduction möglich
- n_dims_95 ≈ embedding_dim: Alle Dimensionen werden genutzt
- Flacher Verlauf: Informationen gleichmäßig verteilt
- Steiler Anfang: Wenige Dimensionen dominieren

**Wann nutzen?**
- ✅ Optimierung von Storage & Compute
- ✅ Bei sehr hochdimensionalen Embeddings (> 1024)
- ✅ Vor Dimension Reduction Experimenten

**Target:**
- n_dims_95 als Basis für Dimension Reduction
- Ziel: < 512 Dimensionen bei gleichbleibender Performance

---

## Distance Distribution Analysis

**Definition:** Verteilung der Distanzen zwischen allen Embedding-Paaren analysieren

**Code:**
```python
# numpy
import numpy as np
import matplotlib.pyplot as plt

# Sample für Performance (bei großen Datasets)
sample_size = 1000
sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
sample_embeddings = embeddings[sample_indices]

# Pairwise Cosine Similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(sample_embeddings)

# Flatten (ohne Diagonale)
triu_indices = np.triu_indices_from(similarities, k=1)
sim_values = similarities[triu_indices]

# Statistiken
print(f"Mean Similarity: {sim_values.mean():.3f}")
print(f"Std Similarity: {sim_values.std():.3f}")
print(f"Min: {sim_values.min():.3f}, Max: {sim_values.max():.3f}")

# Histogram
plt.hist(sim_values, bins=50, edgecolor='black')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Pairwise Similarities')
plt.show()
```

**Interpretation:**
- **Peak bei 0.5-0.7**: Normal - moderate Ähnlichkeit
- **Peak bei > 0.9**: Zu viele sehr ähnliche Embeddings (Duplikate?)
- **Bimodal Distribution**: Zwei klar getrennte Cluster
- **Uniform Distribution**: Embeddings sind zu ähnlich (Model funktioniert nicht)

**Wann nutzen?**
- ✅ Model Evaluation & Selection
- ✅ Debugging von Retrieval-Problemen
- ✅ Data Quality Checks

---

## Embedding Quality Scorecard

| Check | Metric | Target | Pass? |
|-------|--------|--------|-------|
| **Semantic Tests** | Pass Rate | > 80% | ✅/❌ |
| **Clustering** | Silhouette Score | > 0.4 | ✅/❌ |
| **Outliers** | % Outliers | < 5% | ✅/❌ |
| **Cross-Lingual** | Similarity | > 0.7 | ✅/❌ |
| **Dimensionality** | Dims for 95% Var | - | Info |
| **Distribution** | Mean Similarity | 0.4-0.7 | ✅/❌ |

**Overall Assessment:**
- All Pass: Embeddings sind production-ready
- 1-2 Fails: Akzeptabel, aber Monitoring nötig
- > 2 Fails: Model wechseln oder Fine-Tuning erwägen
