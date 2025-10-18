# ProduktRAG - Ein RAG-Lernprojekt

Ein iteratives, lebendes Projekt zum Verstehen von Retrieval-Augmented Generation (RAG). Dies ist ein **Entwicklungstagebuch**, das den kompletten Lernprozess dokumentiert - inklusive Rückschritten, Iterationen und Verbesserungen basierend auf Evaluationsdaten.

## 🎯 Philosophie

- **Iterativ, nicht linear**: Bereits "fertige" Phasen werden basierend auf Evaluationsergebnissen überarbeitet
- **Roadmap-basiert**: Planung in [ROADMAP.md](ROADMAP.md)
- **Diskutieren → Coden → Debuggen**: Ich schreibe, Claude Code unterstützt
- **Prozess > Ergebnis**: Der Lernweg ist wichtiger als perfekter Code

---

## 📔 Entwicklungstagebuch

### 2025-10-18: Datenqualität verbessert, Retrieval-Evaluation erweitert

**Erkenntnisse aus Phase 4:**
- Spezifikationen brauchen mehr Kontext für bessere Embeddings
- Retrieval-Metriken zeigen: Distanz allein reicht nicht

**Änderungen:**
- ↻ **Phase 1 überarbeitet**: Produktnamen werden jetzt in Spec-Header eingebunden (`format_spec()`)
- ↻ **Phase 1-3 neu generiert**: Chunks, Embeddings und Index mit verbesserter Datenqualität
- → **Phase 4 erweitert**: Retrieval-Ergebnisse werden als JSON persistiert, neue Metriken vorbereitet (Cosine Similarity, Dot Product)

**Nächste Schritte:**
- Erweiterte Metrik-Analyse implementieren
- Chunk-Type Distribution analysieren
- Query-Schwierigkeit bewerten

---

### 2025-10-XX: Phase 4 gestartet - Query Retrieval Evaluation

**Was funktioniert:**
- ChromaDB Indexierung liefert erste Ergebnisse
- 61 Test-Queries aus verschiedenen Kategorien

**Erste Probleme erkannt:**
- Spec-Chunks ohne Produktkontext schwer interpretierbar
- Distance-Metrik allein gibt kein vollständiges Bild

→ Führte zur ersten Iteration (siehe oben)

---

### 2025-10-XX: Phasen 1-3 abgeschlossen

- ✅ Phase 1: LLM-basierte Normalisierung mit Mistral
- ✅ Phase 2: GBERT-Embeddings für 1800 Chunks
- ✅ Phase 3: ChromaDB Indexierung

## Technischer Stack

- **Sprache**: Python (Pandas, NumPy, Jupyter Notebooks)
- **Embeddings**: GBERT-large (deutsche Texte)
- **Vector Database**: ChromaDB (lokal)
- **LLM**: Mistral (für Datennormalisierung)
- **Domäne**: Deutsche Produktdaten (Laborkühlschränke, medizinische Geräte)

## Projektstruktur

```
ProduktRAG/
├── 1-normalisation/        # Phase 1: LLM-basierte Datennormalisierung & Chunking
├── 2-embedding/            # Phase 2: GBERT Embeddings generieren
├── 3-indexing/             # Phase 3: ChromaDB Indexierung
├── 4-eval-retrieval/       # Phase 4: Retrieval-Tests (aktuell)
├── 5-eval-model/           # Phase 5: Model-Evaluation (optional)
├── 6-production/           # Phase 6: Production RAG-Pipeline
├── ROADMAP.md              # Detaillierte Projektplanung
└── requirements.txt        # Python Dependencies
```

---

## 📊 Aktueller Stand

| Phase | Status | Iterationen |
|-------|--------|-------------|
| 1. Normalisierung + Chunking | ↻ **Rev. 2** | Produktnamen in Specs |
| 2. Embeddings | ↻ **Rev. 2** | Neu generiert nach Chunk-Update |
| 3. Indexing | ↻ **Rev. 2** | Neu indexiert |
| 4. Retrieval Evaluation | 🎯 **In Arbeit** | Metriken erweitert |
| 5. Model Evaluation | ⏸️ Geplant | - |
| 6. Production Pipeline | ⏸️ Geplant | - |

**Legende:** ✅ Abgeschlossen | ↻ Überarbeitet | 🎯 Aktiv | ⏸️ Ausstehend

Details zur Planung: [ROADMAP.md](ROADMAP.md)

## Setup

```bash
# Dependencies installieren
pip install -r requirements.txt

# Notebooks durchlaufen
jupyter notebook 1-normalisation/1-cleanup.ipynb
```

## 🎓 Lernziele

- **Iterative Entwicklung**: Wie verbessert man ein System basierend auf Metriken?
- **RAG-Pipeline**: Von Rohdaten über Embeddings bis zur semantischen Suche
- **Chunking-Strategien**: Optimierung für Retrieval-Qualität (inkl. Kontextualisierung)
- **Embedding-Konzepte**: Vektorrepräsentationen und Similarity-Metriken
- **Evaluation**: Wie misst und verbessert man Retrieval-Systeme?
- **Python-Praxis**: Pandas, NumPy, ML-Libraries, Jupyter Notebooks

## 📚 Dokumentation

Ausführliche Konzept-Dokumentation befindet sich im separaten Dokumentations-Repository.

---

**Hinweis**: Dies ist ein persönliches Lernprojekt und ein **lebendes Dokument**. Änderungen an früheren Phasen sind Teil des Lernprozesses und werden hier dokumentiert.
