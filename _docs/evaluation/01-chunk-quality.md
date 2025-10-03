# Chunk Quality Evaluation

**Worum geht's?** Datenqualität prüfen BEVOR du embeddest - finde Probleme im Chunking-Prozess.

**Warum wichtig?** Garbage in, garbage out! Schlechte Chunks → Schlechte Embeddings → Schlechtes Retrieval.

---

## 1. Length Statistics

**Definition:** Verteilung der Chunk-Größen analysieren

**Formeln:**
```
Mean = Σ(lengths) / n
Median = middle value (50th percentile)
Std = √(Σ(length - mean)² / n)
IQR = Q3 - Q1  (Interquartile Range)
```

**Code:**
```python
# pandas
import pandas as pd

df = pd.DataFrame(chunks)
lengths = df['text'].str.len()

# Statistiken
print(lengths.describe())
# Output: count, mean, std, min, 25%, 50%, 75%, max
```

**Interpretation:**
- **Mean >> Median**: Wenige sehr lange Chunks (Outliers nach oben)
- **High Std**: Große Varianz in Chunk-Größen
- **Min sehr klein** (< 20): Potentiell leere/nutzlose Chunks
- **Max sehr groß** (> 2000): Chunks zu lang für Context Window

**Wann nutzen?**
- ✅ Nach jedem Chunking-Schritt
- ✅ Beim Vergleich verschiedener Chunking-Strategien
- ✅ Wenn Retrieval-Performance schlecht ist

**Target:**
- Abhängig von Use Case, aber typisch: 200-800 Zeichen
- Std < 50% von Mean (konsistente Größen)

---

## 2. Outlier Detection

**Definition:** Chunks finden die extrem kurz oder lang sind

**Formeln:**
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
Outliers: x < Lower Bound OR x > Upper Bound
```

**Code:**
```python
# pandas
# IQR-Methode (Interquartile Range)
Q1 = lengths.quantile(0.25)
Q3 = lengths.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(lengths < lower_bound) | (lengths > upper_bound)]
print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
```

**Interpretation:**
- **Viele Outliers nach unten**: Chunking splittet zu aggressiv
- **Viele Outliers nach oben**: Chunking splittet nicht genug
- **> 10% Outliers**: Chunking-Strategie überdenken

**Wann nutzen?**
- ✅ Bei unerwarteten Retrieval-Ergebnissen
- ✅ Beim Debugging von Chunking-Logik
- ✅ Vor Production-Deployment

**Action:**
- Zu kurze Chunks: Mergen oder entfernen
- Zu lange Chunks: Weiter splitten

---

## 3. Empty/Null Checks

**Definition:** Ungültige oder leere Chunks identifizieren

**Code:**
```python
# pandas
# Leere Texte
empty = df[df['text'].str.strip() == '']
print(f"Empty: {len(empty)}")

# Null/NaN
null_text = df[df['text'].isna()]
print(f"Null: {len(null_text)}")

# Duplikate
duplicates = df[df.duplicated(subset=['text'], keep=False)]
print(f"Duplicates: {len(duplicates)}")
```

**Interpretation:**
- **Empty Chunks**: Bug im Chunking-Code (z.B. split("\n\n") bei fehlenden Absätzen)
- **Null Values**: Daten-Pipeline Problem (fehlende Felder in Source)
- **Viele Duplikate**: Redundante Informationen (kann gewollt oder Bug sein)

**Wann nutzen?**
- ✅ **Immer** nach Chunking (Basic Sanity Check)
- ✅ Bei Daten aus neuen Quellen
- ✅ Nach Änderungen an Preprocessing

**Target:**
- 0% Empty/Null Chunks
- Duplikate: < 5% (außer bei strukturierten Daten wie Specs)

---

## 4. Metadata Completeness

**Definition:** Prüfen ob alle Chunks vollständige Metadaten haben

**Code:**
```python
# pandas
required_fields = ['id', 'source', 'text']

for field in required_fields:
    missing = df[df[field].isna()]
    pct = len(missing) / len(df) * 100
    print(f"{field}: {pct:.1f}% missing")

# Kategorie-Verteilung
print(df['category'].value_counts())
```

**Interpretation:**
- **Missing IDs**: Tracking/Debugging unmöglich
- **Missing Source**: Kannst nicht zurückverfolgen woher Chunk kommt
- **Ungleiche Kategorie-Verteilung**: Bias im Dataset (z.B. 90% eine Kategorie)

**Wann nutzen?**
- ✅ Bei Metadata-basierten Filtern (Hybrid Retrieval)
- ✅ Für Post-Retrieval Reranking
- ✅ Analytics & Debugging

**Target:**
- 0% missing required fields
- Kategorie-Verteilung sollte Use Case widerspiegeln

---

## 5. Content Quality Patterns

**Definition:** Inhaltliche Probleme durch Pattern-Matching finden

**Code:**
```python
# pandas + regex
# Platzhalter-Text
placeholders = df[df['text'].str.contains(
    r'lorem ipsum|placeholder|TODO|XXX',
    case=False,
    regex=True,
    na=False
)]
print(f"Placeholders: {len(placeholders)}")

# Zu viele Sonderzeichen (korrupte Daten)
special_chars = df['text'].str.count(r'[^\w\s\-.,;:()]')
high_special = df[special_chars > 50]
print(f"High special chars: {len(high_special)}")

# Nur Großbuchstaben (ungeparste Rohdaten)
all_upper = df[df['text'].str.isupper() & (df['text'].str.len() > 30)]
print(f"All uppercase: {len(all_upper)}")
```

**Interpretation:**
- **Platzhalter**: Unfertige Daten in Production gelandet
- **Viele Sonderzeichen**: Encoding-Probleme oder HTML-Tags nicht entfernt
- **Nur Großbuchstaben**: OCR/PDF-Extraktion nicht richtig prozessiert

**Wann nutzen?**
- ✅ Bei Daten aus externen Quellen (Scraping, PDFs, OCR)
- ✅ Nach großen Daten-Updates
- ✅ Wenn User ungewöhnliche Retrieval-Ergebnisse melden

**Target:** < 1% mit Qualitätsproblemen

---

## 📊 Chunk Quality Scorecard

**Quick Checklist:**

| Check | Target | Critical? |
|-------|--------|-----------|
| Empty/Null Chunks | 0% | ✅ Yes |
| Outliers (IQR) | < 10% | ⚠️ Medium |
| Duplicates | < 5% | ⚠️ Medium |
| Missing Metadata | 0% | ✅ Yes |
| Placeholders | 0% | ✅ Yes |
| Encoding Issues | < 1% | ⚠️ Medium |
| Length Std/Mean | < 0.5 | ❌ Nice-to-have |

**Formel: Quality Score**
```
Quality Score = 100 - (total_issues / total_chunks * 100)
Range: [0, 100], höher = besser
```

**Code:**
```python
# pandas
def chunk_quality_score(df):
    issues = 0
    issues += len(df[df['text'].str.strip() == ''])  # Empty
    issues += df['text'].isna().sum()  # Null
    issues += len(df[df['text'].str.len() < 20])  # Too short

    score = max(0, 100 - (issues / len(df) * 100))
    return score

print(f"Quality Score: {chunk_quality_score(df):.1f}/100")
```

**Target Quality Score:** > 95/100
