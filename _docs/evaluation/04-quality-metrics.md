# Quality Metrics

**Warum wichtig?** Die besten Retrieval-Ergebnisse sind wertlos, wenn das LLM falsche oder irrelevante Antworten generiert. Quality Metrics bewerten die tatsächliche Antwortqualität.

---

## Faithfulness / Groundedness

**Definition:** Basiert die Antwort nur auf dem Context (kein Halluzinieren)?

**Measurement:**
```python
# Methode 1: LLM-as-Judge
judge_prompt = """
Context: {context}
Answer: {answer}

Task: Prüfe ob die Antwort NUR auf Informationen aus dem Context basiert.

Bewertung:
- 1.0: Vollständig grounded (alle Fakten im Context)
- 0.5: Teilweise grounded (einige Fakten fehlen im Context)
- 0.0: Halluziniert (Fakten nicht im Context)

Output: Score (0.0-1.0) + Begründung
"""

# Methode 2: RAGAS Framework
# ragas
from ragas.metrics import faithfulness
score = faithfulness.score(
    question=query,
    answer=generated_answer,
    contexts=retrieved_chunks
)
```

**Interpretation:**
- Score > 0.95: Exzellent - keine Halluzinationen
- Score 0.80-0.95: Gut - minimale Abweichungen
- Score < 0.80: Problematisch - LLM erfindet Fakten

**Wann nutzen?**
- ✅ **Kritisch** bei faktischen Anwendungen (Medizin, Recht, Technik)
- ✅ Wenn Vertrauen in Antworten wichtig ist
- ✅ Compliance & Audit-Anforderungen

**Target:** > 0.95 (95% faithful)

**Beispiele:**
```python
# ✅ Faithful
Context: "HMFvh 4001 hat 172 kWh Jahresverbrauch"
Answer: "Der Energieverbrauch beträgt 172 kWh pro Jahr."

# ❌ Hallucination
Context: "HMFvh 4001 hat 172 kWh Jahresverbrauch"
Answer: "Der Energieverbrauch beträgt 150 kWh und ist sehr effizient."
         # 150 steht nicht im Context!
```

---

## Answer Relevance

**Definition:** Beantwortet die Antwort die gestellte Frage?

**Measurement:**
```python
judge_prompt = """
Query: {query}
Answer: {answer}

Bewertet wie relevant die Antwort zur Frage ist:
- 5: Perfekte Antwort (vollständig und direkt)
- 4: Gute Antwort (relevant, aber könnte präziser sein)
- 3: Okay (teilweise relevant)
- 2: Schwach relevant
- 1: Nicht relevant

Score (1-5):
"""
```

**Interpretation:**
- Score 5: Perfekt - beantwortet Frage direkt und vollständig
- Score 3-4: Akzeptabel - Antwort ist relevant aber nicht ideal
- Score 1-2: Schlecht - Antwort verfehlt die Frage

**Wann nutzen?**
- ✅ Jede RAG-Anwendung (Grundmetrik)
- ✅ User Experience wichtig
- ✅ A/B Testing verschiedener Prompts

**Target:** > 4.0 average

**Beispiele:**
```python
# ✅ Relevant (Score: 5)
Query: "Wie hoch ist der Energieverbrauch?"
Answer: "Der Energieverbrauch beträgt 172 kWh pro Jahr."

# ⚠️ Teilweise relevant (Score: 3)
Query: "Wie hoch ist der Energieverbrauch?"
Answer: "Das Gerät hat verschiedene Energiesparfunktionen..."
         # Beantwortet Frage nicht direkt

# ❌ Nicht relevant (Score: 1)
Query: "Wie hoch ist der Energieverbrauch?"
Answer: "Das Gerät hat eine Glastür und 8 Schubfächer."
         # Komplett andere Info
```

---

## Correctness / Factuality

**Definition:** Sind die Fakten in der Antwort korrekt?

**Measurement:**
```python
# Methode 1: Against Ground Truth
# python
def factual_accuracy(answer, ground_truth):
    """
    Vergleiche generierte Antwort mit manuell verifizierten Fakten
    """
    # Mit LLM oder manuell
    pass

# Methode 2: Fact Extraction + Verification
facts_in_answer = extract_facts(answer)
facts_in_context = extract_facts(context)

incorrect_facts = [f for f in facts_in_answer
                   if f not in facts_in_context]
accuracy = 1 - (len(incorrect_facts) / len(facts_in_answer))
```

**Interpretation:**
- 100% Accuracy: Alle Fakten sind korrekt
- 90-99%: Sehr gut - minimale Fehler
- < 90%: Problematisch - zu viele Fehler

**Wann nutzen?**
- ✅ Wenn Ground Truth verfügbar ist
- ✅ Kritische Anwendungen (keine Fehler tolerierbar)
- ✅ Benchmark & Vergleich verschiedener Systeme

**Target:** > 0.90 (90% korrekt)

---

## Completeness

**Definition:** Sind alle wichtigen Aspekte aus dem Context in der Antwort enthalten?

**Measurement:**
```python
judge_prompt = """
Query: {query}
Context: {context}
Answer: {answer}

Bewerte ob alle relevanten Informationen aus dem Context
in der Antwort enthalten sind:

- 5: Vollständig (alle wichtigen Punkte erwähnt)
- 4: Fast vollständig (ein unwichtiger Punkt fehlt)
- 3: Unvollständig (wichtige Punkte fehlen)
- 2: Sehr unvollständig
- 1: Extrem unvollständig

Score (1-5):
"""
```

**Interpretation:**
- Score 5: Alle wichtigen Informationen enthalten
- Score 3-4: Gute Zusammenfassung, kleinere Lücken
- Score 1-2: Wichtige Informationen fehlen

**Wann nutzen?**
- ✅ Wenn vollständige Antworten wichtig sind
- ✅ Komplexe Queries mit mehreren Aspekten
- ✅ Dokumentationssysteme

**Target:** > 3.5

**Beispiel:**
```python
Query: "Was sind die Sicherheitsfeatures?"
Context: """
- Optische und akustische Alarme
- Netzausfallalarm (12h batteriegepuffert)
- Elektronisches Schloss
- DIN 13277 zertifiziert
"""

# ✅ Vollständig (Score: 5)
Answer: "Sicherheitsfeatures: optische/akustische Alarme,
         12h Netzausfallalarm, elektronisches Schloss, DIN 13277."

# ⚠️ Unvollständig (Score: 3)
Answer: "Das Gerät hat Alarme und ein elektronisches Schloss."
         # Netzausfallalarm und DIN fehlen
```

---

## Conciseness

**Definition:** Ist die Antwort prägnant oder zu langatmig?

**Measurement:**
```python
# python
def conciseness_score(answer, target_length=100):
    """
    Einfache Metrik basierend auf Wortanzahl
    """
    word_count = len(answer.split())

    if word_count <= target_length:
        return 1.0
    else:
        # Penalty für zu lange Antworten
        return target_length / word_count

# Mit LLM-Judge:
judge_prompt = """
Answer: {answer}

Ist die Antwort prägnant oder zu weitschweifig?
- 5: Perfekt prägnant
- 3: Okay, etwas langatmig
- 1: Viel zu lang

Score:
"""
```

**Interpretation:**
- Score 1.0 (oder 5): Optimal prägnant
- Score 0.5-0.8 (oder 3): Akzeptabel, etwas lang
- Score < 0.5 (oder 1): Zu weitschweifig

**Wann nutzen?**
- ✅ Chat-Interfaces (User will schnelle Antworten)
- ✅ Mobile Apps (wenig Screen Space)
- ✅ Wenn Kürze wichtig ist

**Target:** 50-150 Wörter (je nach Query-Komplexität)

---

## Citation Accuracy

**Definition:** Sind Quellenangaben korrekt?

**Measurement:**
```python
# python
def citation_accuracy(answer, contexts):
    """
    Prüfe ob zitierte Chunks tatsächlich die Aussagen enthalten
    """
    # Beispiel: Answer enthält [Chunk 3]
    citations = extract_citations(answer)

    correct = 0
    for citation in citations:
        chunk_text = contexts[citation.chunk_id]
        if citation.claim in chunk_text:
            correct += 1

    return correct / len(citations)
```

**Interpretation:**
- 100%: Alle Citations korrekt
- 80-99%: Gut, kleine Fehler
- < 80%: Problematisch - falsche Quellenangaben

**Wann nutzen?**
- ✅ Wenn Answer Citations enthalten soll (`[1]`, `[Chunk 3]`)
- ✅ Transparenz für User wichtig
- ✅ Wissenschaftliche/akademische Anwendungen

**Target:** > 0.95

---

## Quality Metrics Priorität

**Must-Have:**
1. **Faithfulness** - Kein Halluzinieren!
2. **Answer Relevance** - Beantwortet Frage?
3. **Correctness** - Fakten richtig?

**Nice-to-Have:**
4. **Completeness** - Vollständig?
5. **Conciseness** - Prägnant?
6. **Citation Accuracy** - Quellen korrekt? (wenn applicable)

---

## Kombinierte Bewertung

```python
# python
def overall_quality_score(metrics):
    """
    Gewichteter Score über alle Quality Metrics
    """
    weights = {
        'faithfulness': 0.40,      # Kritisch!
        'relevance': 0.30,
        'correctness': 0.20,
        'completeness': 0.10
    }

    score = sum(metrics[m] * weights[m] for m in weights)
    return score

# Beispiel
metrics = {
    'faithfulness': 0.95,
    'relevance': 0.90,
    'correctness': 0.92,
    'completeness': 0.85
}

quality = overall_quality_score(metrics)
print(f"Overall Quality: {quality:.2f}")  # 0.92
```

**Target:** Overall Quality > 0.85
