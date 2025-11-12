# ProduktRAG - Ein RAG-Lernprojekt

## Was ist das hier?

Ein **Lernprojekt** zum Aufbau eines **RAG-Systems (Retrieval-Augmented Generation)** von Grund auf. Ziel ist es, die einzelnen Komponenten einer RAG-Pipeline zu verstehen und hands-on zu implementieren.

**Status:** 🚧 Work in Progress - Retrieval-Evaluation läuft, Reranking-Setup in Arbeit

**Use Case (zum Lernen):** Produktkatalog-Suche mit semantischem Retrieval
- Technische Spezifikationen finden
- Feature-Vergleiche durchführen
- Komplexe Anfragen beantworten

## Daten

**Rohdaten:** ~150 Produktbeschreibungen aus einem Webprojekt
- Strukturierte Daten (JSON)
- Produktbeschreibungen (unstrukturiert)
- Technische Spezifikationen (Key-Value-Paare)

**Anreicherung:** LLM-basierte Aufbereitung (Mistral API)
- Normalisierung von Beschreibungstexten mit Synonym-Ergänzung
- Umwandlung technischer Specs in natürliche Sprache für besseres Embedding
- Kategorie-Extraktion und Produktnamen-Bereinigung (SEO-Suffixe entfernen)
- Anti-Halluzination-Techniken in Prompts

## Pipeline-Struktur

```
data/
├── raw/                    # Rohdaten (products_raw.json, products_raw.jsonl)
├── processed/              # Angereicherte Daten (products_enriched.json, products_chunked.jsonl)
├── prompts/                # System-Prompts und JSON-Schemas für LLM-Agents
└── tests/                  # Generierte Testfragen (specs_question.json, multi_question.json)

notebooks/
├── 1-data_preparation.ipynb    # LLM-basierte Datenanreicherung
├── 2a-chunking.ipynb           # Dokumenten-Chunking
├── 2b-test-generating.ipynb    # Test-Query-Generierung
├── 3-embedding.ipynb           # Embedding-Generierung
├── 4-indexing.ipynb            # ChromaDB-Indexierung
├── 5-retrieval.ipynb           # Retrieval-Evaluation
└── 6-generation.ipynb          # LLM-basierte Antwortgenerierung
```

## Projekt-Organisation

Das Projekt folgt der klassischen RAG-Pipeline:

1. **Data Preparation** → Rohdaten normalisieren und mit LLM anreichern
2. **Chunking** → Dokumente in semantische Einheiten zerlegen
3. **Embedding** → Text in Vektoren umwandeln
4. **Indexing** → Vektoren in ChromaDB speichern
5. **Retrieval** → Relevante Chunks zu Queries finden
6. **Generation** → LLM generiert Antworten basierend auf Retrieved Chunks

**Evaluation:** Retrieval-Performance wird mit LLM-generierten Test-Queries gemessen:
- Single-Chunk-Fragen (gezielt für eine Spec)
- Multi-Chunk-Fragen (benötigen mehrere Chunks zur Beantwortung)
- Metriken: Recall@k, MRR (Mean Reciprocal Rank)

## Aktuelle Herausforderungen & Learnings

### LLM Prompt Engineering
- **Problem:** LLM nutzt Beispiele aus Prompts als Fallback statt eigene Inhalte zu generieren
- **Lösung:** Abstrakte Platzhalter (`[Hersteller]`, `[Modell]`) statt konkreter Beispiele
- **Problem:** Produktnamen enthielten SEO-Kategorien (z.B. "Kirsch LABO-288 Laborkühlschrank")
- **Lösung:** Explizite Bereinigungsregeln + separate Kategorie-Extraktion

### JSON Schema Enforcement
- **Problem:** Mistral API gibt JSON manchmal mit Markdown Code Fences zurück (` ```json ... ``` `)
- **Lösung:** Pydantic-Schemas verwenden statt raten 
- **Lerneffekt:** Dokumentation lesen, nicht nur API-Specs! 📚

### Retrieval-Qualität & Embedding-Optimierung
- **Problem:** Vollständige Sätze mit viel Kontext verschlechtern das Retrieval
- **Erkenntnis:** Mehr natürliche Sprache = mehr grammatikalisches Rauschen im Embedding
- **Lösung:** Telegrafischer Stil für Specs (`[HERSTELLER] [MODELL]: [ATTRIBUT] [WERT]`)
  - Maximale Informationsdichte, minimale Syntax
  - Fast jedes Token ist informationstragend
  - **Resultat:** Recall@10 verbessert von ~50% auf 78% durch optimierte Chunk-Formulierung
- **Problem:** Subtyp-Verwechslung bei ähnlichen Modellnummern (z.B. "8201" vs "8211")
- **Erkenntnis:** Embeddings können sehr ähnliche Zahlenfolgen nicht gut unterscheiden
  - Token-Overlap: ["82", "01"] vs ["82", "11"] → nur 1 Token unterschiedlich
  - Semantische Distanz zu gering für zuverlässige Unterscheidung
- **Optimierungsansatz: Reranking mit Cross-Encoder**
  - Two-Stage Retrieval: Bi-Encoder (schnell, ~20 Kandidaten) → Cross-Encoder (genau, Top-10)
  - Modell: BAAI/bge-reranker-v2-m3 (multilingual, optimiert für kurze Texte)
  - Cross-Encoder vergleicht Query + Dokument direkt (höhere Präzision als Embedding-Distanz)
  - **Status:** Implementierung vorbereitet, Performance-Testing ausstehend (RAM-Constraints)
  - **Nächste Schritte:** Reranking auf separatem Notebook oder kleineres Modell testen
- **Testing:** _[Baseline-Evaluation läuft, Reranking-Vergleich folgt]_

### Data Engineering & Quelldaten-Optimierung
- **Learning:** Strukturierte Quelldaten sind besser handhabbar als unstrukturierte Texte
- **Preprocessing wichtig:** Saubere Normalisierung (Einheiten, Produktnamen) vor LLM-Verarbeitung
- **LLM-Agents gezielt einsetzen:** Specs-Agent für strukturierte Normalisierung + NL-Generierung

## Tech Stack
- **LLM:** Mistral API (mistral-medium-2508) mit JSON Schema
- **Embeddings:** deepset/gbert-large (Bi-Encoder)
- **Reranking:** BAAI/bge-reranker-v2-m3 (Cross-Encoder)
- **Vector Store:** ChromaDB
- **Dev:** Python, Jupyter Notebooks
