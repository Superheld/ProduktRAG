# ProduktRAG - Ein RAG-Lernprojekt

Ein hands-on Projekt zum Verstehen von Retrieval-Augmented Generation (RAG) und zur Vertiefung von Python-Kenntnissen.

## Motivation

Dieses Projekt ist ein **Lernprojekt** auf dem Weg zum Verständnis neuronaler Netze. RAG dient als praktischer Einstieg, um Konzepte wie Embeddings, Vector Search und semantische Ähnlichkeit zu verstehen.

## Arbeitsweise

- **Roadmap-basiert**: Alle Schritte werden in [ROADMAP.md](ROADMAP.md) geplant und dokumentiert
- **Diskutieren → Coden → Debuggen**: Ich schreibe den Code selbst, Claude Code unterstützt beim Debuggen
- **Prozess steht im Vordergrund**: Ausführliche Dokumentation des Lernprozesses

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

## Aktueller Status

✅ **Phase 1**: Chunking-Strategie mit hierarchischen Chunks (Overview, Description, Specs)
✅ **Phase 2**: 1800 GBERT-Embeddings generiert und validiert
✅ **Phase 3**: ChromaDB Indexierung abgeschlossen
🎯 **Phase 4**: Retrieval Evaluation (Next)

Details siehe [ROADMAP.md](ROADMAP.md)

## Setup

```bash
# Dependencies installieren
pip install -r requirements.txt

# Notebooks durchlaufen
jupyter notebook 1-normalisation/1-cleanup.ipynb
```

## Lernziele

- **RAG-Pipeline verstehen**: Von Rohdaten bis zur semantischen Suche
- **Embedding-Konzepte**: Wie funktionieren Vektorrepräsentationen?
- **Chunking-Strategien**: Wie strukturiert man Daten für optimale Retrieval-Qualität?
- **Python-Praxis**: Pandas, NumPy, ML-Libraries
- **Evaluierung**: Wie misst man Qualität von Retrieval-Systemen?

## Dokumentation

Ausführliche Konzept-Dokumentation befindet sich im separaten Dokumentations-Repository.

---

**Hinweis**: Dies ist ein persönliches Lernprojekt. Der Code ist bewusst mit vielen Kommentaren und Erklärungen versehen, um den Lernprozess nachvollziehbar zu machen.
