# ProduktRAG Implementation Roadmap

## Projekt-Neustart: Fundamentale Überarbeitung 🔄

**Status:** Kompletter Neuaufbau mit verbesserter Architektur
**Grund:** Erste Version hatte strukturelle Probleme bei Datenqualität und Chunking-Strategie
**Aktueller Fokus:** Saubere Normalisierung als Fundament für hochwertiges RAG-System

---

## Aktuelle Architektur-Entscheidungen

### Chunk-Strategie (NEU)
Drei-Chunk-Hierarchie für optimales Retrieval:

**1. Overview-Chunk (1x pro Produkt)**
- Zusammenfassung vom LLM generiert
- Enthält Kategorie + Hersteller in Metadata
- Für breite Produkt-Queries

**2. Description-Chunks (mehrere pro Produkt)**
- LLM-bereinigte Absätze ohne Marketing/Firmengeschichte
- Semantisch kohärente Einheiten
- Für feature-spezifische Queries

**3. Spec-Group-Chunks (mehrere pro Produkt)**
- Gruppiert nach Kategorie (Abmessungen, Gewicht, etc.)
- Normalisierte Einheiten (cm, kg, l)
- Strukturierte Keys für Filtering
- Für technische Multi-Kriterien-Suche

### LLM-Agenten-Pipeline (NEU)

**Summary-Agent:**
- Input: Komplettes Produkt
- Output: `{summary, category, manufacturer}`
- Generiert Overview-Chunk

**Description-Agent:**
- Input: Rohe Produktbeschreibung
- Output: Array von bereinigten Absätzen
- Entfernt Marketing, Herstellergeschichte, Redundanz

**Specs-Agent:**
- Input: Array von raw specs
- Output: Gruppierte, normalisierte Specs
- Deutsche Gruppennamen mit Einheit: `"Abmessungen-cm": {"Außenmaße_Breite": 67, ...}`
- Automatische Umrechnung: g→kg, ml→l, mm→cm

### Warum diese Änderungen?

**Problem der alten Version:**
- ❌ Leere/sehr kurze Chunks (z.B. "## Kirsch")
- ❌ Zu granulare Specs (jede einzeln → schlechtes Multi-Kriterien-Retrieval)
- ❌ Marketing-Noise in Descriptions
- ❌ Inkonsistente Einheiten
- ❌ Keine hierarchische Struktur

**Vorteile der neuen Architektur:**
- ✅ Overview für schnellen Produkt-Überblick
- ✅ Saubere, fokussierte Description-Chunks
- ✅ Gruppierte Specs für besseres Matching
- ✅ Normalisierte Daten (cm, kg, l statt mm/cm/m, g/kg, ml/l)
- ✅ Strukturierte Metadata für Hybrid-Search

---

## Current Status

### ✅ COMPLETED: Infrastructure & Tooling
- Agent-Prompts definiert (Summary, Description, Specs)
- LLM-Pipeline mit Mistral implementiert
- Pandas-basierte Datenverarbeitung
- JSONL-Output-Format für Chunks

### 🔄 IN PROGRESS: Phase 1 - Data Normalization & Chunking

**Aktueller Stand:**
- ✅ Raw-Data vorhanden (152 Produkte)
- ✅ Agent-Prompts finalisiert und getestet
- ✅ Code-Struktur für 3-Agenten-Pipeline
- 🔄 Full-Run durch alle 152 Produkte (in Arbeit)
- 📋 Qualitätskontrolle der Agent-Outputs

**Output-Struktur:**
```
1-normalisation/
├── overview_chunks.jsonl      # 152 Chunks (1 pro Produkt)
├── description_chunks.jsonl   # ~600-800 Chunks (mehrere pro Produkt)
└── specs_chunks.jsonl         # ~1200-1500 Chunks (9 Kategorien x ~150 Produkte)
```

**Erwartete Chunk-Anzahl:** ~2000-2500 (vs. alte 4618 mit vielen schlechten)

### 📋 TODO: Phase 2 - Embedding Generation

**Ziel:** Chunks in Vektoren umwandeln mit solidem deutschen Model

**Model-Auswahl:**
- **Start:** `intfloat/multilingual-e5-large`
  - Robustes Multilingual-Model mit sehr guter Performance
  - Gut für deutsche technische Fachsprache
  - 1024 Dimensionen
  - Schnell genug für ~2500 Chunks

**Tasks:**
1. Model laden und vorbereiten
2. Batch-Processing aller normalisierten Chunks (summs + descs + specs)
3. Normalisierung der Embeddings (L2-Normalisierung)
4. Speicherung als `.npy` für schnellen Load
5. Qualitätschecks (keine NaNs, korrekte Dimensionen)

**Expected Output:**
```
2-embedding/
├── embeddings_e5_large.npy          # ~20-30MB
├── chunks_combined.jsonl            # Alle Chunks in einem File
└── embedding_metadata.json          # Model-Info, Timestamp, Dimensionen
```

**Dimensions:** [~2500, 1024]

### 📋 TODO: Phase 3 - Vector Database Integration

**Technology:** ChromaDB (einfach, lokal, perfekt für Lernen)

**Setup:**
1. ChromaDB installieren
2. Collection erstellen mit e5-large-Embeddings
3. Alle Chunks mit Metadata laden
4. Index aufbauen

**Schema:**
```python
chunk = {
    "document": "Abmessungen (cm) - Außenmaße Breite: 67, Tiefe: 72, Höhe: 132",
    "embedding": [...],  # 1024-dim vector
    "metadata": {
        "chunk_type": "overview|description|specs",
        "product_id": "LABO-288",
        "product_title": "Kirsch LABO-288 PRO-ACTIVE",
        "product_url": "https://...",
        "product_category": "Laborkühlschrank",  # Nur bei overview
        "product_manufacturer": "Kirsch",         # Nur bei overview
        "spec_category": "Abmessungen-cm",        # Nur bei specs
        "specs": {"Außenmaße_Breite": 67, ...}   # Nur bei specs
    }
}
```

**Output:**
```
3-indexing/
├── chroma_db/                       # ChromaDB Persistenz
├── setup_index.ipynb                # Setup-Code
└── test_queries.ipynb               # Erste Query-Tests
```

### 📋 TODO: Phase 4 - Retrieval Evaluation

**Ziel:** Testen ob Chunking-Strategie und Embeddings funktionieren

**Test-Queries:**
```python
test_queries = {
    'overview': "Welche Laborkühlschränke von Liebherr gibt es?",
    'technical_specs': "Kühlschrank mit 280L und max. 150cm Höhe",
    'features': "Wie funktioniert die Temperaturüberwachung?",
    'hybrid': "Energieeffizienter Medikamentenkühlschrank mit SmartMonitoring"
}
```

**Evaluation-Metriken:**
- **Precision@K** - Wie viele der Top-K sind relevant?
- **Recall@K** - Wie viele relevante Docs wurden gefunden?
- **MRR** - Position des ersten relevanten Dokuments
- **NDCG** - Ranking-Qualität

**Success Criteria:**
- Precision@3: >80%
- Recall@5: >70%
- MRR: >0.8
- Overview-Chunks ranken hoch bei Produkt-Queries
- Spec-Chunks ranken hoch bei technischen Multi-Kriterien-Queries
- Description-Chunks ranken hoch bei Feature-Queries

**Falls schlecht:** → Phase 5 (Model Evaluation)

### 📋 TODO: Phase 5 - Model Evaluation (Optional)

**Wann:** Nur falls Phase 4 (Retrieval Evaluation) schlechte Ergebnisse zeigt

**Ziel:** Besseres Embedding-Model für deutsche medizinische Fachsprache finden

**Models to Test:**
- `deepset/gbert-large` - Deutsche BERT-Variante
- `GerMedBERT/medbert-512` - Medizin-spezifisch
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Weitere nach Bedarf

**Evaluation-Methoden:**
1. **Semantic Similarity Tests** - Fachbegriffs-Paare (z.B. "Medikamentenkühlschrank" ↔ "Pharmazeutische Lagerung")
2. **Retrieval-Tests** - Dieselben Test-Queries aus Phase 4
3. **Performance-Metrics** - Embedding-Speed, Memory-Usage

**Success Criteria:**
- Bessere Metrics als e5-large in Phase 4
- Gute Differenzierung zwischen technischen Begriffen
- Robustheit gegenüber Synonymen
- Performant genug für Produktion (<2s für 2500 Chunks)

**Output:**
- Falls besseres Model gefunden → Phase 2 & 3 wiederholen
- Sonst → Weiter zu Phase 6

### 📋 TODO: Phase 6 - Production RAG Pipeline

**Components:**
1. Query Interface (FastAPI oder Streamlit)
2. Query → Embedding
3. Vector Search (ChromaDB)
4. Context Assembly (Top-K Chunks + Metadata)
5. LLM Integration (Claude/GPT/Mistral)
6. Response Generation
7. Caching + Logging

**Advanced Features (später):**
- Hybrid Search: Semantic + Structured Filtering
- Multi-Stage Retrieval: Overview → Details
- Metadata-basierte Pre-/Post-Filtering
- Re-Ranking

---

## Neue Chunk-Schema Dokumentation

### Overview-Chunk
```json
{
  "document": "Der Kirsch LABO-288 ist ein Laborkühlschrank mit 280L Volumen...",
  "metadata": {
    "chunk_type": "overview",
    "product_id": "LABO-288",
    "product_title": "Kirsch LABO-288 PRO-ACTIVE Laborkühlschrank",
    "product_url": "https://...",
    "product_category": "Laborkühlschrank",
    "product_manufacturer": "Kirsch"
  }
}
```

### Description-Chunk
```json
{
  "document": "Die elektronische Temperatursteuerung regelt die Temperatur...",
  "metadata": {
    "chunk_type": "description",
    "product_id": "LABO-288",
    "product_title": "Kirsch LABO-288 PRO-ACTIVE Laborkühlschrank",
    "product_url": "https://..."
  }
}
```

### Specs-Chunk
```json
{
  "document": "Abmessungen (cm) - Außenmaße Breite: 67, Tiefe: 72, Höhe: 132",
  "metadata": {
    "chunk_type": "specs",
    "spec_category": "Abmessungen-cm",
    "product_id": "LABO-288",
    "product_title": "Kirsch LABO-288 PRO-ACTIVE Laborkühlschrank",
    "product_url": "https://...",
    "specs": {
      "Außenmaße_Breite": 67,
      "Außenmaße_Tiefe": 72,
      "Außenmaße_Höhe": 132
    }
  }
}
```

---

## Updated File Structure

```
ProduktRAG/
├── _docs/                              # 📚 Documentation
│   ├── CHUNKING-DEEP-DIVE.md          # ✅ Umfassender Chunking-Guide
│   └── EMBEDDING-STRATEGIES.md        # ✅ Embedding-Best-Practices
├── 1-chunking/                        # 🔄 IN PROGRESS - Normalisierung + Chunking
│   ├── agent_description.md           # ✅ Description-Agent Prompt
│   ├── agent_specs.md                 # ✅ Specs-Agent Prompt
│   ├── agent_summary.md               # ✅ Summary-Agent Prompt
│   ├── 1-cleanup.ipynb                # 🔄 LLM-Pipeline Implementation
│   ├── products_raw.json              # ✅ Raw data (152 Produkte)
│   ├── summs_chunks.jsonl             # 📋 TODO - Overview chunks
│   ├── descs_chunks.jsonl             # 📋 TODO - Description chunks
│   └── specs_chunks.jsonl             # 📋 TODO - Specs chunks
├── 2-embedding/                       # 📋 TODO - Embedding generation
│   ├── generate_embeddings.ipynb      # Embedding-Pipeline
│   ├── embeddings_e5_large.npy        # Vector-Ausgabe
│   ├── chunks_combined.jsonl          # Alle Chunks kombiniert
│   └── embedding_metadata.json        # Model-Info
├── 3-indexing/                        # 📋 TODO - ChromaDB setup
│   ├── chroma_db/                     # ChromaDB Persistenz
│   ├── setup_index.ipynb              # Index erstellen
│   └── test_queries.ipynb             # Erste Tests
├── 4-eval-retrieval/                  # 📋 TODO - Retrieval evaluation
│   ├── eval_retrieval.ipynb           # Metrics berechnen
│   └── test_queries.json              # Test-Query-Set
├── 5-eval-model/                      # 📋 TODO OPTIONAL - Model comparison
│   ├── compare_models.ipynb           # Model-Vergleich
│   └── results.json                   # Evaluation-Results
├── 6-production/                      # 📋 TODO - Production RAG
│   ├── app.py                         # FastAPI/Streamlit
│   ├── rag_pipeline.py                # RAG-Logik
│   └── config.yaml                    # Konfiguration
├── ROADMAP.md                         # ✅ This file
└── requirements.txt                   # ✅ Dependencies
```

---

## Lessons Learned (aus Version 1)

### Was schiefging:
- **Chunking zu früh:** Ohne Datenbereinigung → viele schlechte Chunks
- **Kein LLM für Normalisierung:** Manuelle Regex-Spielchen ineffektiv
- **Zu granulare Specs:** Jede Spec einzeln → schlechtes Multi-Kriterien-Matching
- **Keine Hierarchie:** Alle Chunks gleich behandelt

### Was wir jetzt besser machen:
- **LLM-First Approach:** Lass das LLM die Arbeit machen
- **Hierarchische Chunks:** Overview, Details, Specs für verschiedene Query-Typen
- **Strukturierte Normalisierung:** Einheiten vereinheitlicht, gruppiert
- **Quality First:** Lieber 2500 gute Chunks als 4618 mit Noise

### Nächste Schritte nach Chunking:
1. **Embedding** - Mit multilingual-e5-large starten
2. **Indexing** - ChromaDB aufsetzen
3. **Retrieval Evaluation** - Testen ob's funktioniert
4. **Model Evaluation** - Nur falls nötig, andere Models probieren
5. **Production** - RAG-Pipeline bauen

---

## Performance Targets

### Data Quality (Phase 1):
- ✅ Alle Chunks >10 Zeichen
- ✅ Keine Marketing-Floskeln in Descriptions
- ✅ Konsistente Einheiten (cm, kg, l)
- ✅ Strukturierte Gruppierung

### Embedding Performance (Phase 2):
- Embedding-Zeit für 2500 Chunks: <2min
- Embedding-Dimensionen: 1024
- Keine NaNs oder Inf-Werte

### Retrieval Quality (Phase 4):
- Precision@3: >80%
- Recall@5: >70%
- MRR: >0.8
- NDCG: >0.75

### Production Performance (Phase 6):
- Query Latency: <3s (Embedding + Retrieval + Generation)
- Memory Usage: <4GB
- Scalability: 100+ concurrent queries

---

## Documentation

Siehe `_docs/` Ordner für:
- **CHUNKING-DEEP-DIVE.md** - Warum, Was, Wie? (Universell einsetzbar)
- **RAG-EMBEDDING-STRATEGIES.md** - Best Practices für Embeddings

---

*Last updated: 2025-10-03*
*Status: Phase 1 (Chunking) in Arbeit - 60% Complete*

---

## Quick Reference: Pipeline-Überblick

```
Phase 1: Chunking ✅ (in Arbeit)
   ↓
Phase 2: Embedding (multilingual-e5-large)
   ↓
Phase 3: Indexing (ChromaDB)
   ↓
Phase 4: Retrieval Evaluation (testen!)
   ↓
Phase 5: Model Evaluation (nur falls Phase 4 schlecht)
   ↓
Phase 6: Production RAG
```
