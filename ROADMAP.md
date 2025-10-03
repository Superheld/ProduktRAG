# ProduktRAG Implementation Roadmap

## Projekt-Neustart: Fundamentale Überarbeitung 🔄

**Status:** Kompletter Neuaufbau mit verbesserter Architektur
**Grund:** Erste Version hatte strukturelle Probleme bei Datenqualität und Chunking-Strategie
**Aktueller Fokus:** Saubere Normalisierung als Fundament für hochwertiges RAG-System

## Current Status

### ✅ COMPLETED: Infrastructure & Tooling
- Agent-Prompts definiert (Summary, Description, Specs)
- LLM-Pipeline mit Mistral implementiert
- Pandas-basierte Datenverarbeitung
- JSONL-Output-Format für Chunks

### ✅ COMPLETED: Phase 1 - Data Normalization & Chunking

**Erledigt:**
- ✅ Raw-Data bereinigt (leere Produkte entfernt)
- ✅ Drei LLM-Agenten implementiert und getestet:
  - **summs_agent.md**: Generiert Zusammenfassung + extrahiert Category/Manufacturer
  - **descs_agent.md**: Bereinigt Produktbeschreibungen von Marketing-Floskeln
  - **specs_agent.md**: Normalisiert technische Specs mit zwei Input-Formaten (Array + Text-Fallback)
- ✅ Notebook refactored:
  - `chunk_id` für alle Chunks (Format: `{product_id}_{type}_{index}`)
  - `base_metadata` mit spread operator (`**`) für DRY-Code
  - Fallback-Handling für Produkte ohne Specs
  - `enumerate()` für Index-Tracking
- ✅ Schema-Verbesserungen:
  - specs_agent verarbeitet jetzt Array-Input UND Text-Input konsistent
  - Alle 12 Spec-Gruppen immer im Output (auch wenn leer)
  - Verstärkte Prompt-Regeln für konsistentes Output-Schema

**Output-Struktur:**
```
1-normalisation/
├── summs_chunks.jsonl      # Summary chunks (1 pro Produkt)
├── descs_chunks.jsonl      # Description chunks (mehrere pro Produkt)
└── specs_chunks.jsonl      # Specs chunks (gruppiert nach Kategorien)
```

**Key Learnings:**
- LLM benötigt sehr explizite Schema-Definitionen (Beispiele > abstrakte Regeln)
- Fallback-Handling wichtig für fehlende/inkonsistente Daten
- Metadata-Struktur sollte früh finalisiert werden (chunk_id, base_metadata)

### 🎯 NEXT: Phase 2 - Embedding Generation

**Ziel:** Chunks in Vektoren umwandeln mit solidem deutschen Model

**Model-Auswahl:**
- **Start:** `intfloat/multilingual-e5-large`
  - Robustes Multilingual-Model mit sehr guter Performance
  - Gut für deutsche technische Fachsprache
  - 1024 Dimensionen
  - Schnell genug für alle Chunks

**Tasks:**
1. ✅ Chunks aus Phase 1 vorhanden (summs_chunks.jsonl, descs_chunks.jsonl, specs_chunks.jsonl)
2. 📋 Model laden und vorbereiten (`sentence-transformers`)
3. 📋 Batch-Processing aller Chunks:
   - Alle drei JSONL-Files kombinieren
   - Embedding für jedes `document` Field generieren
   - L2-Normalisierung anwenden
4. 📋 Speicherung:
   - Embeddings als `.npy` Array
   - Chunks mit IDs als `.jsonl` (für Mapping)
   - Metadata-File mit Model-Info
5. 📋 Qualitätschecks:
   - Keine NaNs/Inf-Werte
   - Korrekte Dimensionen [N, 1024]
   - Semantische Plausibilität (ähnliche Produkte → ähnliche Vektoren)

**Expected Output:**
```
2-embedding/
├── embeddings_e5_large.npy          # Vector array [N, 1024]
├── chunks_combined.jsonl            # Alle Chunks mit chunk_id
└── embedding_metadata.json          # Model name, dimensions, timestamp
```

**Nächste Schritte nach Fertigstellung:**
- Embedding-Qualität visuell prüfen (t-SNE/UMAP)
- Zur Phase 3 (Indexing) übergehen

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
*Status: Phase 1 (Chunking) ✅ Complete | Phase 2 (Embedding) 🎯 Next*

---

## Quick Reference: Pipeline-Überblick

```
Phase 1: Chunking ✅ DONE
   ↓
Phase 2: Embedding 🎯 NEXT (multilingual-e5-large)
   ↓
Phase 3: Indexing (ChromaDB)
   ↓
Phase 4: Retrieval Evaluation (testen!)
   ↓
Phase 5: Model Evaluation (nur falls Phase 4 schlecht)
   ↓
Phase 6: Production RAG
```

### Phase 1 Summary (Completed)
- 3 LLM-Agenten für Normalisierung (summs, descs, specs)
- Chunk-Schema mit chunk_id und strukturierter Metadata
- Specs-Agent mit Dual-Input-Support (Array + Text)
- Bereinigtes Dataset ohne leere Produkte
- Output: 3x JSONL-Files (summs, descs, specs)
