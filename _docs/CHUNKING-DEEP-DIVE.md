# Chunking Deep Dive: Warum, Was, Wie?

## Was ist Chunking und warum brauchen wir es?

**Problem:**
- Embedding-Modelle haben Input-Limits (meist 512 Tokens)
- Lange Dokumente passen nicht komplett rein
- Zu viel Information in einem Embedding → verwässerte Semantik

**Lösung: Chunking**
- Dokumente in kleinere Einheiten aufteilen
- Jeder Chunk bekommt sein eigenes Embedding
- Bei Suche: Finde die **relevantesten Chunks**, nicht ganze Dokumente

## Warum ist Chunking kritisch für RAG?

### 1. **Retrieval-Genauigkeit**
```
Schlechter Chunk: "Liebherr bietet seit 60 Jahren Kühlschränke.
                   Der LABO-288 hat 280l Volumen und kostet 3000€."

Guter Chunk: "Der Liebherr LABO-288 Laborkühlschrank hat ein
              Nutzvolumen von 280 Litern."
```

**Query:** "Kühlschrank mit 300 Liter Volumen"
- Schlechter Chunk: Viel Noise (60 Jahre, Preis) → schlechtere Similarity
- Guter Chunk: Fokussiert auf eine Info → bessere Similarity

### 2. **Kontext-Qualität für LLM**
Nach Retrieval werden die Top-K Chunks an das LLM gegeben:
- **Zu kleine Chunks:** LLM bekommt fragmentierte Info
- **Zu große Chunks:** Irrelevante Info verschwendet Context-Window

### 3. **Embedding-Qualität**
Embedding-Modelle funktionieren am besten wenn:
- Ein Chunk = Eine kohärente Idee
- Semantisch zusammenhängende Konzepte
- Keine vermischten Themen

## Chunking-Strategien im Detail

### 1. Fixed-Size Chunking
**Wie:** Feste Anzahl Tokens/Zeichen
```python
chunk_size = 512
overlap = 50
chunks = split_by_tokens(text, chunk_size, overlap)
```

**Vorteile:**
- Einfach zu implementieren
- Vorhersagbare Chunk-Größen
- Gleichmäßige Embedding-Verteilung

**Nachteile:**
- Schneidet mitten im Satz/Konzept ab
- Keine semantische Kohärenz
- Overlap kann Redundanz erzeugen

**Wann nutzen:**
- Unstrukturierte Fließtexte (Romane, Artikel)
- Wenn Semantik weniger wichtig ist

### 2. Semantic Chunking (Paragraph-based)
**Wie:** Nach natürlichen Grenzen (Absätze, Sätze)
```python
paragraphs = text.split('\n\n')
chunks = [p for p in paragraphs if len(p) > min_length]
```

**Vorteile:**
- Behält semantische Einheiten
- Natürliche Kohärenz
- Lesbar und verständlich

**Nachteile:**
- Variable Chunk-Größen
- Manche Paragraphen zu lang/kurz
- Abhängig von Dokumentstruktur

**Wann nutzen:**
- Gut strukturierte Texte
- Wenn Absätze thematisch getrennt sind

### 3. Recursive Chunking
**Wie:** Hierarchisch aufteilen bis Zielgröße erreicht
```python
1. Versuche Split bei \n\n (Paragraphen)
2. Falls zu groß → Split bei \n (Zeilen)
3. Falls zu groß → Split bei . (Sätze)
4. Falls zu groß → Split bei Wörtern
```

**Vorteile:**
- Balance zwischen Semantik und Größe
- Flexible Anpassung
- Behält so viel Kontext wie möglich

**Nachteile:**
- Komplexer zu implementieren
- Inkonsistente Chunk-Typen

**Wann nutzen:**
- Gemischte Dokumenttypen
- Wenn Fixed-Size zu grob ist

### 4. Document-Structure Based
**Wie:** Nach strukturellen Elementen (Headings, Listen)
```python
# Markdown-basiert
chunks = split_by_headings(markdown_text)

# HTML-basiert
chunks = extract_by_tags(html, ['h2', 'section'])
```

**Vorteile:**
- Nutzt vorhandene Struktur
- Chunks haben klare Themen
- Perfekt für Dokumentationen

**Nachteile:**
- Braucht strukturierte Docs
- Variable Größen
- Nicht für Fließtext

**Wann nutzen:**
- Markdown/HTML Docs
- Technische Dokumentationen
- Produktbeschreibungen mit strukturierten Headings

### 5. Sentence-Based
**Wie:** Kombiniere Sätze bis max_length
```python
sentences = nltk.sent_tokenize(text)
chunks = combine_sentences_to_chunks(sentences, max_length=512)
```

**Vorteile:**
- Keine abgeschnittenen Sätze
- Semantisch sinnvoll
- Kontrollierbare Größe

**Nachteile:**
- Braucht NLP-Library (spaCy, NLTK)
- Langsamer
- Kann Sätze trennen die zusammengehören

**Wann nutzen:**
- Fließtext ohne Struktur
- Wenn Satzgrenzen wichtig sind

## Chunk-Schema: Was gehört in einen Chunk?

### Minimales Schema
```json
{
  "document": "Der tatsächliche Text-Chunk",
  "metadata": {
    "source": "Woher kommt der Chunk?"
  }
}
```

### Erweitertes Schema (besser!)
```json
{
  "document": "Das Produkt XY-500 hat ein Volumen von 280 Litern.",
  "metadata": {
    "id": "XY-500",
    "title": "Produktname XY-500",
    "url": "https://...",
    "chunk_type": "description",
    "section": "Technische Daten",
    "category": "Kategorie A"
  }
}
```

### Warum Metadata wichtig ist:

**1. Post-Retrieval Filtering**
```python
# Finde Chunks, dann filtere nach Kategorie
results = retrieve(query, top_k=20)
filtered = [r for r in results if r.metadata['category'] == 'Kategorie A']
```

**2. Dokument-Identifikation**
Nach Retrieval musst du wissen: Aus welchem Dokument/Item kommt dieser Chunk?
→ Metadata hat `id` und `title`

**3. Context für LLM**
```
[Metadata: Produkt XY-500 - Technische Daten]
Das Gerät hat ein Volumen von 280 Litern.
```
→ LLM weiß mehr Context als nur der Chunk-Text

**4. Debugging & Analytics**
- Welche Chunk-Typen performen gut?
- Aus welcher Section kommen die meisten Treffer?

## Anwendungsfälle & Best Practices

### Use Case 1: E-Commerce Produktkataloge

**Herausforderung:** Strukturierte Produktdaten (Beschreibungen + Spezifikationen)

**Empfohlene Strategie:**
- **Descriptions:** Semantic Chunking (Paragraphen)
- **Specs:** Gruppierung nach Kategorie (z.B. alle Abmessungen zusammen)

```python
# Descriptions: Paragraph-based
paragraphs = product['description'].split('\n\n')
chunks = [p for p in paragraphs if len(p) > 50]

# Specs: Gruppiert nach Kategorie
dimension_specs = [s for s in specs if 'dimension' in s['category']]
chunk = {
    'document': f"Dimensions: {', '.join([f'{s['key']}: {s['value']}' for s in dimension_specs])}",
    'metadata': {'product_id': ..., 'chunk_type': 'specs_dimensions'}
}
```

**Häufige Fehler:**
- ❌ Jede Spec einzeln → zu granular (User sucht oft mehrere Kriterien)
- ❌ Sehr kurze Marketing-Snippets als Chunks
- ✅ Specs nach Thema gruppieren (Abmessungen, Leistung, etc.)
- ✅ Marketing-Text vom Fachtext trennen

### Use Case 2: Technische Dokumentationen

**Herausforderung:** Hierarchische Struktur, Code-Beispiele, lange Dokumente

**Empfohlene Strategie:**
- Document-Structure Based (nach Headings)
- Recursive Chunking für zu lange Sections

```python
# Nach Markdown-Struktur
sections = split_by_headings(markdown_doc)

for section in sections:
    if len(section) > max_length:
        # Recursive Split
        subsections = split_by_subheadings(section)

    chunk = {
        'document': section,
        'metadata': {
            'heading': extract_heading(section),
            'level': heading_level,  # h1, h2, h3
            'parent_heading': parent_section
        }
    }
```

**Häufige Fehler:**
- ❌ Fixed-Size über Code-Blöcke hinweg (zerstört Syntax)
- ❌ Headings ohne Content als Chunk
- ✅ Code-Beispiele mit Erklärung zusammen halten
- ✅ Cross-References in Metadata speichern

### Use Case 3: Customer Support / FAQ

**Herausforderung:** Frage-Antwort-Paare, kurze Dokumente

**Empfohlene Strategie:**
- Sentence-Based für Fließtext
- Q&A-Pairs als einzelne Chunks

```python
# FAQ: Eine Frage + Antwort = Ein Chunk
chunk = {
    'document': f"Q: {question}\nA: {answer}",
    'metadata': {
        'chunk_type': 'faq',
        'category': 'billing',
        'keywords': ['payment', 'invoice']
    }
}

# Support-Artikel: Sentence-Based
sentences = nltk.sent_tokenize(article)
chunks = combine_sentences(sentences, target_length=200)
```

**Häufige Fehler:**
- ❌ Nur Antworten ohne Fragen (verschlechtert Matching)
- ❌ Zu lange Support-Artikel als ein Chunk
- ✅ Frage im Chunk für besseres Query-Matching
- ✅ Synonyme/Keywords in Metadata

## Allgemeine Best Practices

### 1. Chunk-Size Guidelines
- **Min:** 50 Tokens (zu kurz = kein Context)
- **Max:** 512 Tokens (typisches Embedding-Model Limit)
- **Sweet Spot:** 100-300 Tokens

### 2. Metadata-Schema
```json
{
  "document": "...",
  "metadata": {
    "id": "doc-123",
    "title": "Document Title",
    "category": "Category A",
    "chunk_type": "description|spec|faq",
    "section": "Section Name",
    "url": "https://..."
  }
}
```

### 3. Quality Checks
```python
# Nach Chunking validieren:
assert len(chunk['document']) > 10  # Nicht zu kurz
assert len(chunk['document']) < 2000  # Nicht zu lang
assert 'id' in chunk['metadata']  # Metadata vollständig
```

### 4. Chunk-Overlap
```python
# Manchmal hilfreich wenn Konzepte über Paragraphen gehen
chunks_with_overlap = []
for i, para in enumerate(paragraphs):
    chunk_text = para
    if i > 0:  # Füge letzten Satz vom vorherigen Paragraph hinzu
        chunk_text = last_sentence(paragraphs[i-1]) + " " + para
    chunks_with_overlap.append(chunk_text)
```

### 5. Hybrid-Strategien
Kombiniere verschiedene Ansätze für unterschiedliche Dokumenttypen:
- Text-Paragraphen → Semantic Chunking
- Strukturierte Daten → Grouped by Category
- Code-Beispiele → Keep together
- Tabellen → Als einzelne Chunks

## Evaluation

### 1. Chunk-Statistiken
```python
chunk_lengths = [len(c['document']) for c in chunks]
print(f"Avg: {np.mean(chunk_lengths)}")
print(f"Min: {np.min(chunk_lengths)}")
print(f"Max: {np.max(chunk_lengths)}")
print(f"Std: {np.std(chunk_lengths)}")
```

### 2. Retrieval-Tests
```python
# Teste konkrete Queries
query = "Product with 280 liter volume"
results = retrieve(query, top_k=5)

# Sind die Top-5 Chunks relevant?
# Enthalten sie die Info um die Frage zu beantworten?
```

### 3. Coverage-Check
```python
# Werden alle wichtigen Infos in Chunks gecovered?
original_doc = document['text']
chunked_text = " ".join([c['document'] for c in chunks])

# Ist wichtige Info verloren gegangen?
assert "important_feature" in chunked_text
```

Für detaillierte Evaluation-Methoden siehe [Evaluation Guides](./evaluation/00-overview.md)

## Zusammenfassung

**Chunking ist kritisch** weil es direkt beeinflusst:
- Retrieval-Genauigkeit
- Embedding-Qualität
- LLM-Context-Qualität

**Wähle Strategie basierend auf:**
- Dokumentstruktur (strukturiert vs. Fließtext)
- Use Case (Produkte, Docs, FAQ)
- Query-Patterns (spezifisch vs. breit)

**Best Practices:**
- 100-300 Tokens pro Chunk
- Semantische Kohärenz wahren
- Gute Metadata für Filtering
- Hybrid-Ansätze für gemischte Daten
- Immer evaluieren und iterieren!
