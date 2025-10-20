# ProduktRAG - Ein RAG-Lernprojekt

## Was ist das hier?

Ein **Lernprojekt** zum Aufbau eines **RAG-Systems (Retrieval-Augmented Generation)** von Grund auf. Ziel ist es, die einzelnen Komponenten einer RAG-Pipeline zu verstehen und hands-on zu implementieren.

**Status:** 🚧 Work in Progress - Iteratives Refactoring basierend auf Evaluationsergebnissen

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
- Normalisierung von Beschreibungstexten
- Umwandlung in natürliche Sprache für besseres Embedding
- Metadaten-Extraktion

## Pipeline-Struktur

```
data/
├── raw/                    # Rohdaten (products_raw.json)
├── processed/              # Angereicherte Daten (products_enriched.json)
└── promts/                 # System-Prompts und Schemas für LLM-Agents

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

**Evaluation:** Retrieval-Performance wird mit manuell kuratierten Test-Queries (Ground Truth) gemessen.
