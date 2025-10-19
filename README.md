# ProduktRAG - Ein RAG-Lernprojekt

Ein iteratives, lebendes Projekt zum Verstehen von Retrieval-Augmented Generation (RAG). Dies ist ein **Entwicklungstagebuch**, das den kompletten Lernprozess dokumentiert - inklusive Rückschritten, Iterationen und Verbesserungen basierend auf Evaluationsdaten.

## 🎯 Philosophie

- **Iterativ, nicht linear**: Bereits "fertige" Phasen werden basierend auf Evaluationsergebnissen überarbeitet
- **Diskutieren → Coden → Debuggen**: Ich schreibe, Claude Code unterstützt
- **Prozess > Ergebnis**: Der Lernweg ist wichtiger als perfekter Code

---

## 📔 Entwicklungstagebuch

### 2025-10-19: Pipeline optimieren

Die Roadmap ist geschichte :-) Das Ziel ist ja klar. Beim bearbeiten der Daten fällt mir auf, das die Pipeline nicht ordendlich strukturiert ist. Das möchte ich direkt zum optimieren nutzen, was sich sowieso abgezeichnet hat. Die Daten werden auf das beschränkt was für das Projekt notwendig ist und vielleicht irgendwann erweitert.

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