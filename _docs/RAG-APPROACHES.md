# RAG-Ansätze: Von Basic bis Advanced

Eine Übersicht über verschiedene RAG (Retrieval-Augmented Generation) Ansätze, sortiert nach Komplexität.

---

## 1. Basic RAG (Naive RAG)

**Konzept:** Query → Retrieve → Generate

```python
# Das Einfachste
query = "Wie viele Schubfächer?"
docs = vector_db.search(query, top_k=5)
context = "\n".join(docs)
answer = llm(f"Context: {context}\n\nFrage: {query}")
```

**Gut für:**
- FAQ-Systeme
- Simple Dokumentation
- Prototypen

**Probleme:**
- Keine Kontext-Optimierung
- Retrieval-Fehler = falsche Antwort
- Keine Quellenangaben

**Komplexität:** ⭐ (sehr einfach)

---

## 2. RAG mit Metadata Filtering

**Konzept:** Erst filtern, dann suchen

```python
# Filter nach Produkt-Kategorie
results = vector_db.search(
    query,
    filter={"category": "Medikamentenkühlschrank"}
)
```

**Gut für:**
- Multi-Tenant Systeme
- Produktkataloge
- Wenn Sie strukturierte Daten haben

**Vorteil:**
- Weniger irrelevante Ergebnisse
- Schneller (kleinerer Search Space)
- Bessere Precision

**Komplexität:** ⭐⭐ (einfach)

**Implementierung:**
```python
# Vector DB mit Metadata
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
client.upsert(
    collection_name="products",
    points=[{
        "id": i,
        "vector": embedding,
        "payload": {
            "product_id": "HMFvh-4001",
            "category": "Kühlschrank",
            "type": "spec"
        }
    }]
)

# Suche mit Filter
results = client.search(
    collection_name="products",
    query_vector=query_embedding,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "Kühlschrank"}},
            {"key": "type", "match": {"value": "spec"}}
        ]
    }
)
```

---

## 3. Hybrid Search

**Konzept:** Dense + Sparse kombinieren

```python
# BM25 (Keywords) + Dense (Semantik)
bm25_results = bm25.search(query)
dense_results = vector_db.search(query)

# Fusion (z.B. Reciprocal Rank Fusion)
final = combine(bm25_results, dense_results, alpha=0.5)
```

**Reciprocal Rank Fusion (RRF):**
```python
def reciprocal_rank_fusion(results_list, k=60):
    """
    Kombiniert mehrere Ranking-Listen
    """
    scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Gut für:**
- Technische Dokumentation
- Mix aus exakten Begriffen + Semantik
- Medizinische/Legale Texte

**Wann wichtig:**
- Fachbegriffe, Produktnummern, Normen
- "DIN 13277" muss exakt matchen
- "8 Schubfächer" vs. "Wie viele Fächer?"

**Komplexität:** ⭐⭐⭐ (mittel)

---

## 4. Query Transformation

### a) Query Rewriting

**Konzept:** LLM verbessert die Query

```python
original = "Wieviel Strom verbraucht das?"

prompt = f"""
Verbessere diese Suchanfrage für eine technische Datenbank:
Query: "{original}"

Formuliere präziser und verwende Fachbegriffe.
Antworte nur mit der verbesserten Query.
"""

rewritten = llm(prompt)
# → "Energieverbrauch pro Jahr in kWh"

results = search(rewritten)
```

**Variante: HyDE (Hypothetical Document Embeddings):**
```python
# LLM generiert hypothetische Antwort
hypothetical_doc = llm(f"Beantworte: {query}")

# Suche mit dem hypothetischen Dokument statt der Query
results = vector_db.search(hypothetical_doc)
```

### b) Query Decomposition

**Konzept:** Komplexe Query in Sub-Queries aufteilen

```python
query = "Energieeffizienter Kühlschrank mit Alarm unter 200 kWh"

prompt = f"""
Zerlege diese komplexe Suchanfrage in einfache Sub-Queries:
"{query}"

Format: JSON-Array von Strings
"""

sub_queries = llm(prompt)
# → ["Energieverbrauch < 200 kWh", "Alarmsystem vorhanden", "Kühlschrank Modelle"]

# Jede einzeln suchen
all_results = []
for sq in sub_queries:
    results = search(sq)
    all_results.extend(results)

# Kombinieren und deduplizieren
final = deduplicate_and_rank(all_results)
```

### c) Multi-Query

**Konzept:** Mehrere Varianten der gleichen Frage

```python
prompt = f"""
Generiere 3 unterschiedliche Formulierungen dieser Frage:
"{query}"

1. Formale Fachsprache
2. Umgangssprachlich
3. Mit Synonymen

Format: JSON-Array
"""

queries = llm(prompt)
# → ["Energiekonsumption", "Stromverbrauch", "kWh pro Jahr"]

results = [search(q) for q in queries]
merged = reciprocal_rank_fusion(results)
```

**Gut für:**
- Komplexe User-Fragen
- Mehrdeutige Queries
- Bessere Recall

**Komplexität:** ⭐⭐⭐ (mittel, LLM-Calls)

---

## 5. Re-Ranking

**Konzept:** Erst grob suchen (100 Docs), dann fein ranken (Top 5)

```python
# Stage 1: Fast Retrieval (billig)
candidates = vector_db.search(query, top_k=100)

# Stage 2: Re-Ranking (teuer, aber präzise)
# Option A: Cross-Encoder
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

pairs = [(query, doc) for doc in candidates]
scores = cross_encoder.predict(pairs)
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
final_docs = [doc for doc, score in ranked[:5]]

# Option B: LLM Re-Ranking
prompt = f"""
Query: {query}

Dokumente:
{format_docs(candidates)}

Ranke die Dokumente nach Relevanz (1-100).
Format: JSON mit doc_id und score
"""
reranked = llm(prompt)
```

**Models für Re-Ranking:**
- **Cross-Encoders:** `ms-marco-MiniLM`, `bge-reranker-large`
- **LLMs:** GPT-4, Claude (teuer aber sehr gut)
- **Spezialisiert:** Cohere Rerank API

**Gut für:**
- Wenn Precision wichtiger als Latenz
- Große Dokumentmengen
- Medizinische/Legale Anwendungen

**Performance-Vergleich:**
| Methode | Latenz | Qualität | Kosten |
|---------|--------|----------|--------|
| Nur Bi-Encoder | 50ms | 70% | € |
| + Cross-Encoder | 200ms | 85% | € |
| + LLM Re-Rank | 2s | 95% | €€€ |

**Komplexität:** ⭐⭐⭐ (mittel)

---

## 6. Hierarchical RAG (Parent-Child)

**Konzept:** Kleine Chunks für Retrieval, große für Context

```python
# Indexiere kleine Chunks (für Suche)
small_chunks = [
    {"id": "chunk_1", "text": "8 Schubfächer", "parent_id": "para_1"},
    {"id": "chunk_2", "text": "Energieverbrauch 172 kWh", "parent_id": "para_1"},
]

# Parent Dokumente (größerer Kontext)
parent_docs = {
    "para_1": "Der HMFvh 4001 verfügt über 8 Schubfächer und verbraucht 172 kWh/Jahr. Mit SmartMonitoring..."
}

# Retrieval
retrieved_small = search(query)  # Findet "chunk_1"

# Aber gib dem LLM den ganzen Paragraph
parent_id = retrieved_small["parent_id"]
full_context = parent_docs[parent_id]

answer = llm(context=full_context, query=query)
```

**Varianten:**

**A) Sentence Window:**
```python
# Indexiere Sätze, gib aber ±2 Sätze als Kontext
sentences = split_into_sentences(doc)
retrieved_sentence = search(query)  # Satz 5
context = sentences[3:8]  # Sätze 3-7 (Window um Satz 5)
```

**B) Auto-Merging:**
```python
# Hierarchie: Dokument → Paragraphen → Sätze
# Wenn mehrere Sätze aus gleichem Paragraph gefunden werden → merge zu Paragraph
retrieved = [sent_1, sent_2, sent_3]
if all_from_same_paragraph(retrieved):
    context = get_paragraph(retrieved)
else:
    context = retrieved
```

**Gut für:**
- Lange Dokumente
- Wenn Kontext wichtig ist
- Vermeidet Fragmentierung

**Problem, das es löst:**
```
Query: "Wie funktioniert SafetyDevice?"

❌ Ohne Hierarchical:
Chunk 1: "SafetyDevice aktiviert sich bei Störung"
Chunk 2: "Es stabilisiert die Temperatur"
Chunk 3: "Schutz vor Einfrieren"
→ Fragmentiert, kein Zusammenhang

✅ Mit Hierarchical:
Parent Doc: "Bei einem Stromausfall wird das SafetyDevice aktiviert.
Es stabilisiert die Temperatur und schützt den Inhalt vor dem Einfrieren.
Der Alarm wird ausgelöst..."
→ Vollständiger Kontext
```

**Komplexität:** ⭐⭐⭐ (mittel, mehr Datenstrukturen)

---

## 7. Fusion RAG

**Konzept:** Mehrere Retrieval-Strategien parallel

```python
# Parallele Suchen
results_dense = dense_search(query)
results_sparse = bm25_search(query)
results_llm_expanded = search(llm_expand(query))

# Reciprocal Rank Fusion
final = reciprocal_rank_fusion([
    results_dense,
    results_sparse,
    results_llm_expanded
])
```

**Implementierung:**
```python
import asyncio

async def fusion_search(query):
    # Parallele Ausführung
    results = await asyncio.gather(
        async_dense_search(query),
        async_bm25_search(query),
        async_expanded_search(llm_expand(query))
    )

    # Fusion mit Gewichtung
    weights = [0.5, 0.3, 0.2]  # Dense wichtiger
    return weighted_fusion(results, weights)
```

**Strategien für Fusion:**

**A) Reciprocal Rank Fusion (RRF):**
```python
score = sum(1 / (k + rank_i) for rank_i in all_ranks)
```

**B) Distribution-Based Fusion:**
```python
# Normalisiere Scores auf [0,1]
normalized_scores = (scores - min(scores)) / (max(scores) - min(scores))
combined = sum(weight_i * normalized_scores_i)
```

**C) Learned Fusion:**
```python
# ML-Modell lernt optimale Gewichte
fusion_model = train(features=[dense_score, bm25_score, ...], labels=relevance)
final_score = fusion_model.predict([scores...])
```

**Gut für:**
- Maximale Recall
- Wenn ein Ansatz allein nicht reicht
- Production-Systeme mit hohen Anforderungen

**Komplexität:** ⭐⭐⭐⭐ (komplex, viele Komponenten)

---

## 8. Self-RAG (Reflective RAG)

**Konzept:** LLM entscheidet selbst, ob es Retrieval braucht

```python
# 1. LLM versucht zu antworten
prompt_initial = f"Beantworte kurz: {query}"
initial_answer = llm(prompt_initial)

# 2. LLM bewertet eigene Unsicherheit
prompt_confidence = f"""
Frage: {query}
Deine Antwort: {initial_answer}

Bist du dir sicher? Bewerte 0-10:
- 0-3: Unsicher, brauche externe Quellen
- 4-6: Teilweise sicher
- 7-10: Sehr sicher

Antworte nur mit einer Zahl.
"""

confidence = int(llm(prompt_confidence))

# 3. Conditional Retrieval
if confidence < 7:
    docs = retrieve(query)
    final_answer = llm(f"Context: {docs}\n\nFrage: {query}")
else:
    final_answer = initial_answer

return final_answer, {"retrieved": confidence < 7}
```

**Variante: Self-Critique**
```python
answer = llm(query + context)

critique = llm(f"""
Antwort: {answer}
Context: {context}

Ist die Antwort durch den Context gestützt?
Gibt es Widersprüche?
Antworte: OK / NEEDS_REVISION
""")

if critique == "NEEDS_REVISION":
    # Mehr Dokumente holen oder Query neu formulieren
    additional_docs = retrieve(expanded_query)
    answer = llm(query + context + additional_docs)
```

**Gut für:**
- Mix aus parametrischem + retrieval Wissen
- Spart Retrieval-Kosten
- Adaptive Systeme

**Metriken:**
- **Retrieval Rate:** Wie oft wird tatsächlich gesucht?
- **Precision von Self-Assessment:** Wie oft ist die Confidence-Einschätzung korrekt?

**Komplexität:** ⭐⭐⭐⭐ (komplex, LLM-basierte Entscheidungen)

---

## 9. Iterative RAG (Multi-Turn)

**Konzept:** Mehrere Retrieval-Runden mit Verfeinerung

```python
query = "Wie funktioniert SafetyDevice?"

# Round 1: Erste Suche
docs1 = retrieve(query)
answer1 = llm(f"Erkläre kurz: {docs1}")

# Round 2: Follow-up basierend auf Answer1
follow_up_prompt = f"""
Basierend auf dieser Info: {answer1}

Was sind offene Fragen?
Was muss ich noch wissen über SafetyDevice?
"""

follow_up_query = llm(follow_up_prompt)
docs2 = retrieve(follow_up_query)

# Final Answer
final_prompt = f"""
Ursprüngliche Frage: {query}

Information Round 1: {docs1}
Information Round 2: {docs2}

Gib eine vollständige Antwort.
"""

final = llm(final_prompt)
```

**Chain-of-Thought Variante:**
```python
# LLM plant Retrieval-Schritte
plan = llm(f"""
Frage: {query}

Plane Recherche-Schritte:
1. Was muss ich zuerst herausfinden?
2. Was dann?
3. ...

Format: JSON-Array von Schritten
""")

# Führe Plan aus
results = []
for step in plan:
    docs = retrieve(step["query"])
    summary = llm(f"Fasse zusammen: {docs}")
    results.append(summary)

# Kombiniere Ergebnisse
final = llm(f"Synthese aus: {results}\n\nFrage: {query}")
```

**Gut für:**
- Komplexe Fragen
- Multi-Hop Reasoning ("Wer hat X erfunden, und wo wurde diese Person geboren?")
- Research-Tasks

**Problem:**
- Latenz (mehrere LLM + Retrieval Calls)
- Kostenintensiv
- Kann in Loops geraten

**Komplexität:** ⭐⭐⭐⭐ (komplex, State-Management)

---

## 10. Agentic RAG (mit LangGraph etc.)

**Konzept:** LLM ist ein Agent mit Tools

```python
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import Tool

# Definiere Tools
tools = [
    Tool(
        name="search_specs",
        func=search_technical_specs,
        description="Sucht in technischen Spezifikationen (Zahlen, Maße, kWh)"
    ),
    Tool(
        name="search_descriptions",
        func=search_semantic_descriptions,
        description="Sucht in Produktbeschreibungen (Features, Funktionen)"
    ),
    Tool(
        name="calculator",
        func=calculate,
        description="Führt Berechnungen durch"
    ),
]

# Agent erstellen
agent = create_react_agent(llm, tools)

# Agent arbeitet autonom
result = agent.invoke({
    "messages": [("user", "Vergleiche Energieverbrauch von 3 Kühlschränken")]
})

# Interner Ablauf (vom Agent selbst geplant):
# 1. Thought: "Ich brauche Energieverbrauchsdaten"
# 2. Action: search_specs("Energieverbrauch")
# 3. Observation: [172 kWh, 180 kWh, 165 kWh]
# 4. Thought: "Ich kann vergleichen"
# 5. Final Answer: "Modell C ist am effizientesten mit 165 kWh/Jahr..."
```

**LangGraph State Machine:**
```python
from langgraph.graph import StateGraph, END

# Definiere States
workflow = StateGraph()

workflow.add_node("classify_query", classify_query_type)
workflow.add_node("search_technical", search_specs)
workflow.add_node("search_semantic", search_descriptions)
workflow.add_node("synthesize", generate_answer)

# Edges (Übergänge)
workflow.add_conditional_edges(
    "classify_query",
    route_query,
    {
        "technical": "search_technical",
        "semantic": "search_semantic",
        "both": ["search_technical", "search_semantic"]
    }
)

workflow.add_edge("search_technical", "synthesize")
workflow.add_edge("search_semantic", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()
result = app.invoke({"query": user_query})
```

**Gut für:**
- Komplexe Workflows
- Tool-Usage nötig (Calculator, APIs, Web-Search)
- Multi-Step Tasks
- Dynamische Entscheidungen

**Frameworks:**
- **LangGraph:** State Machine, volle Kontrolle
- **AutoGen:** Multi-Agent, Conversation zwischen Agents
- **CrewAI:** Team von Agents mit Rollen

**Komplexität:** ⭐⭐⭐⭐⭐ (sehr komplex, Framework-Knowledge nötig)

---

## 11. GraphRAG

**Konzept:** Knowledge Graph + RAG

### Schritte:

**1. Graph-Erstellung:**
```python
# Extrahiere Entities und Relations aus Dokumenten
from llama_index import KnowledgeGraphIndex

docs = load_documents()

# LLM extrahiert automatisch
kg_index = KnowledgeGraphIndex.from_documents(
    docs,
    llm=llm,
    max_triplets_per_chunk=10
)

# Resultierende Triplets:
# (HMFvh 4001, hat_feature, SafetyDevice)
# (SafetyDevice, schützt_vor, Einfrieren)
# (SafetyDevice, aktiviert_bei, Stromausfall)
# (Stromausfall, löst_aus, Alarm)
```

**2. Graph-Query:**
```python
query = "Was schützt vor Temperaturproblemen?"

# Cypher Query (Neo4j)
cypher = """
MATCH (problem:Problem {name: "Temperaturproblem"})<-[:SCHÜTZT_VOR]-(feature:Feature)
MATCH (feature)<-[:HAT_FEATURE]-(product:Product)
RETURN product.name, feature.name
"""

results = neo4j_db.query(cypher)

# Oder: Subgraph Extraction
subgraph = kg_index.query_subgraph(
    query="Temperaturprobleme",
    depth=2
)

# Subgraph → Text für LLM
context = subgraph.to_text()
answer = llm(f"Context: {context}\n\nFrage: {query}")
```

**3. Microsoft GraphRAG Ansatz:**
```python
# Community Detection auf Graph
communities = detect_communities(knowledge_graph)

# Für jede Community: Zusammenfassung
community_summaries = {}
for community in communities:
    nodes = community.get_nodes()
    summary = llm(f"Fasse diese Konzepte zusammen: {nodes}")
    community_summaries[community.id] = summary

# Query über Community-Summaries (schneller als ganzer Graph)
relevant_communities = search_communities(query)
context = [community_summaries[c] for c in relevant_communities]
answer = llm(context + query)
```

**Gut für:**
- Beziehungen sind wichtig
- Multi-Hop Reasoning ("Welche Features schützen vor Problemen, die durch Stromausfall entstehen?")
- Komplexe Domänen (Medizin, Legal, Supply Chain)
- Wenn Sie bereits strukturierte Daten haben

**Vorteile:**
- Explizite Beziehungen
- Multi-Hop Queries einfach
- Erklärbarkeit (Pfad im Graph zeigen)

**Nachteile:**
- Graph-Erstellung aufwändig
- Braucht Graph-DB (Neo4j, etc.)
- Wartung des Graphs

**Tools:**
- **Neo4j:** Graph Database
- **LlamaIndex:** KnowledgeGraphIndex
- **Microsoft GraphRAG:** Community-basierter Ansatz
- **LangChain:** Neo4jGraph integration

**Komplexität:** ⭐⭐⭐⭐⭐ (sehr komplex, Graph-DB Knowledge nötig)

---

## 12. Corrective RAG (CRAG)

**Konzept:** LLM bewertet Retrieval-Qualität und korrigiert

```python
# Initial Retrieval
docs = retrieve(query)

# Relevance Evaluation
prompt_eval = f"""
Query: {query}

Dokumente:
{format_docs(docs)}

Bewerte jedes Dokument:
- RELEVANT: Beantwortet die Frage direkt
- PARTIALLY_RELEVANT: Enthält verwandte Info
- IRRELEVANT: Nicht hilfreich

Format: JSON mit doc_id und rating
"""

evaluations = llm(prompt_eval)

relevant_docs = [d for d in docs if evaluations[d.id] == "RELEVANT"]
partially_relevant = [d for d in docs if evaluations[d.id] == "PARTIALLY_RELEVANT"]

# Decision Logic
if len(relevant_docs) >= 3:
    # Genug gute Docs
    answer = llm(relevant_docs + query)

elif len(relevant_docs) + len(partially_relevant) >= 2:
    # Web-Search als Ergänzung
    web_docs = google_search(query)
    all_docs = relevant_docs + partially_relevant + web_docs
    answer = llm(all_docs + query)

else:
    # Komplett neu: Query Rewriting
    rewritten = llm(f"Formuliere um für bessere Suche: {query}")
    docs2 = retrieve(rewritten)

    # Wenn immer noch schlecht → Web-only
    if quality_score(docs2) < threshold:
        web_docs = google_search(query)
        answer = llm(web_docs + query)
    else:
        answer = llm(docs2 + query)
```

**Mit Confidence Scores:**
```python
# Retrieval mit Scores
docs_with_scores = retrieve_with_scores(query)

# Adaptive Threshold
if max(scores) > 0.8:
    # High confidence
    top_docs = docs_with_scores[:3]
elif max(scores) > 0.6:
    # Medium confidence → mehr Docs
    top_docs = docs_with_scores[:10]
    reranked = cross_encoder_rerank(query, top_docs)
    top_docs = reranked[:3]
else:
    # Low confidence → Fallback
    web_docs = web_search(query)
    top_docs = web_docs[:3]

answer = llm(top_docs + query)
```

**Gut für:**
- Hohe Qualität-Anforderungen
- Wenn Retrieval-Fehler teuer sind (Medizin, Legal)
- Production-Systeme mit Fallbacks

**Metriken:**
- **Correction Rate:** Wie oft wird korrigiert?
- **Web-Search Rate:** Wie oft Fallback zu Web?
- **Quality Improvement:** Wie viel besser nach Correction?

**Komplexität:** ⭐⭐⭐⭐ (komplex, viele Entscheidungen)

---

## 13. Modular RAG

**Konzept:** Komponenten als austauschbare Module

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
   ┌───▼────────┐
   │   Router   │ ──→ Technical? → BM25
   │            │ ──→ Semantic? → Dense
   │            │ ──→ Hybrid? → Both
   └───┬────────┘
       │
   ┌───▼────────┐
   │ Retriever  │ ← Austauschbar (BM25, Dense, Hybrid)
   └───┬────────┘
       │
   ┌───▼────────┐
   │ Re-Ranker  │ ← Optional
   └───┬────────┘
       │
   ┌───▼────────┐
   │ Generator  │ ← LLM
   └────────────┘
```

**Implementierung:**
```python
from abc import ABC, abstractmethod

# Abstrakte Interfaces
class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int):
        pass

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list):
        pass

# Konkrete Implementierungen
class DenseRetriever(Retriever):
    def retrieve(self, query, top_k):
        embedding = self.model.encode(query)
        return self.vector_db.search(embedding, top_k)

class BM25Retriever(Retriever):
    def retrieve(self, query, top_k):
        return self.bm25.get_top_n(query, self.corpus, top_k)

class HybridRetriever(Retriever):
    def __init__(self, dense, sparse):
        self.dense = dense
        self.sparse = sparse

    def retrieve(self, query, top_k):
        dense_results = self.dense.retrieve(query, top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k * 2)
        return reciprocal_rank_fusion([dense_results, sparse_results])

# RAG Pipeline
class RAGPipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker = None):
        self.retriever = retriever
        self.reranker = reranker

    def query(self, query: str):
        # 1. Retrieve
        docs = self.retriever.retrieve(query, top_k=20)

        # 2. Optional Re-rank
        if self.reranker:
            docs = self.reranker.rerank(query, docs)[:5]

        # 3. Generate
        context = "\n".join([d.text for d in docs])
        answer = self.llm(f"Context: {context}\n\nQuery: {query}")

        return answer, docs

# Einfach austauschen
pipeline_v1 = RAGPipeline(retriever=DenseRetriever())
pipeline_v2 = RAGPipeline(retriever=HybridRetriever(dense, bm25), reranker=CrossEncoderReranker())
```

**A/B Testing:**
```python
# Teste verschiedene Konfigurationen
configs = [
    {"retriever": "dense", "reranker": None},
    {"retriever": "dense", "reranker": "cross_encoder"},
    {"retriever": "hybrid", "reranker": None},
    {"retriever": "hybrid", "reranker": "llm"},
]

results = {}
for config in configs:
    pipeline = build_pipeline(config)
    metrics = evaluate(pipeline, test_queries)
    results[str(config)] = metrics

# Visualisiere
import pandas as pd
df = pd.DataFrame(results).T
df.plot(kind='bar')
```

**Gut für:**
- Flexibilität
- A/B Testing verschiedener Komponenten
- Production-Systeme
- Experimente ohne Code-Rewrites

**Vorteile:**
- Einfach neue Retriever/Reranker austesten
- Clean Code
- Wiederverwendbar

**Komplexität:** ⭐⭐⭐⭐ (komplex, aber gut strukturiert)

---

## 14. RAG + Fine-Tuning

**Konzept:** Kombiniere RAG mit Fine-Tuned LLM

### Varianten:

**A) Fine-Tuned Generator:**
```python
# 1. Fine-tune LLM auf Ihre Domain
training_data = [
    {"context": doc1, "query": q1, "answer": a1},
    {"context": doc2, "query": q2, "answer": a2},
    # ...
]

fine_tuned_llm = fine_tune(
    base_model="llama-2-7b",
    training_data=training_data,
    task="rag_generation"
)

# 2. RAG mit Fine-Tuned LLM
docs = retrieve(query)
answer = fine_tuned_llm(docs + query)  # Bessere Domain-Antworten
```

**B) Fine-Tuned Retriever:**
```python
# Trainiere Embedding-Model auf Ihre Daten
from sentence_transformers import SentenceTransformer, InputExample, losses

# Training Pairs aus Ihren Daten
train_examples = [
    InputExample(texts=["Energieverbrauch", "172 kWh pro Jahr"], label=1.0),
    InputExample(texts=["Schubfächer", "8 Schubladen"], label=1.0),
    InputExample(texts=["Energieverbrauch", "Farbe: Weiß"], label=0.0),
]

model = SentenceTransformer('deepset/gbert-base')
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)

# Verwende fine-tuned model für Embeddings
embeddings = model.encode(chunks)
```

**C) End-to-End RAG Fine-Tuning:**
```python
# Neue Ansätze: RA-DIT, REPLUG
# Trainiere Retriever + Generator zusammen

# Pseudo-Code
for batch in training_data:
    # Forward Pass
    retrieved_docs = retriever(batch.query)
    answer = generator(batch.query, retrieved_docs)

    # Loss berechnet über beide Komponenten
    loss = compute_loss(answer, batch.ground_truth)

    # Backprop durch beide
    loss.backward()
    update_retriever()
    update_generator()
```

**Gut für:**
- Sehr spezifische Domänen (Medizintechnik, Legal)
- Wenn Sie viele Domain-Daten haben
- Production mit höchsten Anforderungen

**Wann sinnvoll:**
- Sie haben >10k annotierte Query-Doc-Answer Triplets
- Domain-spezifische Terminologie
- Off-the-shelf Models funktionieren nicht gut genug

**Aufwand:**
- Datensammlung und Annotation (teuer!)
- GPU-Training
- Evaluation

**Komplexität:** ⭐⭐⭐⭐⭐ (sehr komplex, ML-Engineering)

---

## Vergleichstabelle: Wann welcher Ansatz?

| Use Case | Empfohlener Ansatz | Komplexität | Latenz | Kosten |
|----------|-------------------|-------------|--------|--------|
| **FAQ, Simple Docs** | Basic RAG | ⭐ | 1s | € |
| **Produktkatalog** | Hybrid + Metadata | ⭐⭐⭐ | 2s | € |
| **Technische Doku** | Hybrid + Re-Ranking | ⭐⭐⭐ | 3s | €€ |
| **Customer Support** | Self-RAG + Metadata | ⭐⭐⭐⭐ | 2-5s | €€ |
| **Research** | Iterative RAG | ⭐⭐⭐⭐ | 10s+ | €€€ |
| **Komplexe Workflows** | Agentic RAG | ⭐⭐⭐⭐⭐ | 15s+ | €€€ |
| **Legal/Medical** | CRAG + Re-Ranking | ⭐⭐⭐⭐ | 5s | €€€ |
| **Knowledge-Intensive** | GraphRAG | ⭐⭐⭐⭐⭐ | 3-8s | €€ |
| **High-Volume Production** | Modular RAG | ⭐⭐⭐⭐ | 2s | € |

---

## Für ProduktRAG-Projekt: Empfohlene Evolution

### Phase 1: ✅ Basic RAG (aktuell)
```python
embeddings = gbert.encode(chunks)
results = cosine_similarity(query, embeddings)
```

### Phase 2: 🔄 Hybrid + Metadata (nächster Schritt)
```python
# BM25 für technische Specs
bm25_results = bm25.search(query)

# Dense für Descriptions
dense_results = vector_db.search(query)

# Metadata Filtering
filtered = filter_by_product(results)

# Fusion
final = reciprocal_rank_fusion([bm25_results, dense_results])
```

### Phase 3: 📋 Query Classification (später)
```python
query_type = classify(query)  # technical/semantic/hybrid

if query_type == "technical":
    results = bm25_search(query, collection="specs")
elif query_type == "semantic":
    results = dense_search(query, collection="descriptions")
else:
    results = hybrid_search(query)
```

### Phase 4: 💡 Optional Enhancements
- Re-Ranking mit Cross-Encoder
- Hierarchical (Specs → ganzer Product Context)
- Self-RAG (nur bei unsicheren Fragen retrieval)

### ❌ NICHT empfohlen für Produktkatalog:
- GraphRAG (Overkill, keine komplexen Beziehungen)
- Agentic RAG (zu komplex für simple Product Queries)
- Fine-Tuning (zu aufwändig, GBERT ist bereits gut)

---

## Evaluations-Checkliste

Für jeden Ansatz sollten Sie messen:

### Retrieval-Qualität:
- ✅ **Precision@K** - Sind Top-K relevant?
- ✅ **Recall@K** - Alle relevanten gefunden?
- ✅ **MRR** - Position des ersten relevanten
- ✅ **NDCG** - Ranking-Qualität

### System-Performance:
- ⏱️ **Latenz** - Query → Answer Zeit
- 💰 **Kosten** - LLM API Calls, Compute
- 📊 **Durchsatz** - Queries pro Sekunde

### Answer-Qualität:
- ✅ **Accuracy** - Ist die Antwort korrekt?
- 📚 **Faithfulness** - Basiert auf Context?
- 🎯 **Relevance** - Beantwortet die Frage?

### Business-Metriken:
- 👍 **User Satisfaction** - Ratings
- 🔄 **Retry Rate** - Wie oft neue Query?
- ✅ **Task Completion** - Ziel erreicht?

---

## Ressourcen & Frameworks

### Frameworks:
- **LangChain** - Umfassend, viele Integrationen
- **LlamaIndex** - RAG-fokussiert, einfacher
- **Haystack** - Production-ready, modular
- **LangGraph** - State Machines für Agentic RAG

### Vector Databases:
- **Qdrant** - Open-source, Hybrid Search Support
- **Weaviate** - GraphQL API, Multi-Modal
- **Milvus** - Sehr performant, skaliert gut
- **ChromaDB** - Simpel, lokal, gut für Prototypen

### Evaluation:
- **RAGAS** - RAG-spezifische Metriken
- **TruLens** - Tracing und Evaluation
- **LangSmith** - LangChain Monitoring
- **Phoenix** - Open-source Observability

---

## Abschließende Gedanken

**Start Simple:**
- Basic RAG reicht für 80% der Use Cases
- Erst optimieren, wenn Sie echte Probleme haben
- Messen Sie, bevor Sie komplexer werden

**Add Complexity Only When Needed:**
- Jede zusätzliche Komponente = mehr zu warten
- Mehr Fehlerquellen
- Höhere Latenz

**Focus on Evaluation:**
- Ohne Metriken wissen Sie nicht, ob Verbesserungen helfen
- Ground Truth > fancy Ansätze
- User Feedback ist Gold wert

**Für Ihr Projekt:**
Sie sind auf dem richtigen Weg! Basic → Hybrid → Metadata ist perfekt für Produktkataloge. Alles darüber hinaus ist Nice-to-Have, aber nicht nötig.

---

*Erstellt für: ProduktRAG (Medizintechnik E-Commerce)*
*Datum: 2025-09-30*
*Ziel: Übersicht über RAG-Ansätze von simpel bis advanced*
