# System Metrics

**Warum wichtig?** Die beste RAG-Qualität ist wertlos wenn das System zu langsam ist, zu teuer, oder ständig abstürzt. System Metrics bewerten Production-Performance.

---

## Latency (Response Time)

**Definition:** Zeit von Query bis Answer

**Code:**
```python
# python
import time
import numpy as np

def measure_latency(rag_system, queries, n_runs=10):
    latencies = []

    for query in queries:
        start = time.time()
        answer = rag_system.query(query)
        latency = time.time() - start
        latencies.append(latency)

    return {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'max': np.max(latencies)
    }

# Breakdown
def latency_breakdown(query):
    start = time.time()

    t1 = time.time()
    query_emb = embed_query(query)
    embed_time = time.time() - t1

    t2 = time.time()
    chunks = retrieve_chunks(query_emb, k=5)
    retrieval_time = time.time() - t2

    t3 = time.time()
    answer = llm_generate(query, chunks)
    generation_time = time.time() - t3

    return {
        'total': time.time() - start,
        'embedding': embed_time,
        'retrieval': retrieval_time,
        'generation': generation_time
    }
```

**Interpretation:**
- Total < 2s: Exzellent - interaktive User Experience
- Total 2-5s: Gut - akzeptabel für die meisten Use Cases
- Total > 5s: Problematisch - User werden ungeduldig

**Komponenten-Breakdown:**
- **Embedding**: Meist < 100ms (schnell)
- **Retrieval**: 100-500ms (je nach DB-Größe)
- **Generation**: 1-3s (dominiert meist die Latency)

**Wann nutzen?**
- ✅ **Production Monitoring** (kontinuierlich)
- ✅ Performance Optimization
- ✅ SLA Compliance

**Target:**
- Total: < 3s (End-to-End)
- Embedding: < 0.1s
- Retrieval: < 0.5s
- Generation: < 2s

---

## Throughput

**Definition:** Queries pro Sekunde (QPS)

**Code:**
```python
# python
import time
import random

def measure_throughput(rag_system, queries, duration=60):
    """Misst QPS über duration Sekunden"""
    start = time.time()
    count = 0

    while time.time() - start < duration:
        query = random.choice(queries)
        rag_system.query(query)
        count += 1

    elapsed = time.time() - start
    qps = count / elapsed

    return qps

# Concurrent Throughput
import concurrent.futures

def concurrent_throughput(rag_system, queries, workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        start = time.time()
        futures = [executor.submit(rag_system.query, q)
                   for q in queries]
        concurrent.futures.wait(futures)
        elapsed = time.time() - start

    return len(queries) / elapsed
```

**Interpretation:**
- QPS > 100: High-Performance System
- QPS 10-100: Production-Ready
- QPS < 10: Development/Prototyp

**Wann nutzen?**
- ✅ Load Testing vor Production
- ✅ Capacity Planning
- ✅ Vergleich verschiedener Infrastrukturen

**Target:**
- Development: > 1 QPS
- Production: > 10 QPS
- High-Load: > 100 QPS

---

## Cost per Query

**Definition:** Kosten pro Query in Production

**Formel:**
```
Cost = (Input_Tokens × Price_Input / 1000) + (Output_Tokens × Price_Output / 1000)

Embedding: meist kostenlos (lokal)
Vector DB: meist kostenlos (ChromaDB/Qdrant lokal)
LLM: Pay-per-Token
```

**Code:**
```python
# python
def cost_per_query(config):
    """
    Berechne Kosten basierend auf Usage
    """
    costs = {
        'embedding': 0,  # Lokal = kostenlos
        'vector_db': 0,  # ChromaDB lokal = kostenlos
        'llm_input': config['input_tokens'] * config['price_per_1k_input'] / 1000,
        'llm_output': config['output_tokens'] * config['price_per_1k_output'] / 1000,
    }

    total = sum(costs.values())
    return total, costs

# Beispiel: GPT-4
config = {
    'input_tokens': 1500,  # Query + 5 Chunks à 200 tokens
    'output_tokens': 150,   # Antwort
    'price_per_1k_input': 0.03,
    'price_per_1k_output': 0.06
}

total, breakdown = cost_per_query(config)
print(f"Cost per query: ${total:.4f}")
# ~$0.054 per query

# Monatliche Kosten bei 10k Queries
monthly_cost = total * 10000
print(f"Monthly cost (10k queries): ${monthly_cost:.2f}")
```

**Interpretation:**
- < $0.01/query: Sehr günstig (lokale Models)
- $0.01-$0.10/query: Standard (GPT-3.5, GPT-4)
- > $0.10/query: Teuer (nur für High-Value Use Cases)

**Wann nutzen?**
- ✅ Budget Planning
- ✅ Model Selection (Cost vs Performance)
- ✅ ROI Berechnung

**Target:** < $0.05 per Query

**Optimization:**
- Kleineres LLM (GPT-3.5 vs GPT-4)
- Weniger Chunks (Top-3 statt Top-5)
- Caching häufiger Queries
- Lokale Models (Llama, Mistral)

---

## Cache Hit Rate

**Definition:** Wie oft können Queries aus Cache bedient werden?

**Code:**
```python
# python
class RAGSystemWithCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def query(self, query):
        cache_key = self._normalize(query)

        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        answer = self._generate_answer(query)
        self.cache[cache_key] = answer
        return answer

    def _normalize(self, query):
        """Normalisiere Query für Cache-Lookup"""
        return query.lower().strip()

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def stats(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate(),
            'cache_size': len(self.cache)
        }
```

**Formel:**
```
Hit Rate = Cache_Hits / (Cache_Hits + Cache_Misses)
Range: [0, 1], höher = besser
```

**Interpretation:**
- Hit Rate > 50%: Exzellent - viele wiederkehrende Queries
- Hit Rate 30-50%: Gut - Caching lohnt sich
- Hit Rate < 30%: Caching bringt wenig (viele unique Queries)

**Wann nutzen?**
- ✅ Bei wiederkehrenden Queries (FAQ, Standard-Fragen)
- ✅ Cost Optimization (Cache = kostenlos)
- ✅ Latency Reduction

**Target:** > 30% Hit Rate (je nach Use Case)

**Cache-Strategien:**
- **LRU Cache**: Least Recently Used (begrenzte Größe)
- **TTL Cache**: Time-To-Live (Auto-Expire)
- **Semantic Cache**: Ähnliche Queries → selbe Antwort

---

## Error Rate

**Definition:** Wie oft schlägt das System fehl?

**Code:**
```python
# python
error_types = {
    'no_results': 0,      # Keine Chunks gefunden
    'timeout': 0,         # Query zu langsam
    'llm_error': 0,       # LLM API Error
    'parse_error': 0,     # Antwort nicht parsebar
}

total_queries = 0

def track_error(error_type):
    global total_queries
    error_types[error_type] += 1

def error_rate():
    total_errors = sum(error_types.values())
    return total_errors / total_queries if total_queries > 0 else 0

def error_breakdown():
    total_errors = sum(error_types.values())
    return {
        error_type: count / total_errors
        for error_type, count in error_types.items()
        if count > 0
    }

# Usage
try:
    answer = rag_system.query(query)
    total_queries += 1
except TimeoutError:
    track_error('timeout')
except Exception as e:
    track_error('llm_error')
```

**Formel:**
```
Error Rate = Total_Errors / Total_Queries
Range: [0, 1], niedriger = besser
```

**Interpretation:**
- Error Rate < 1%: Exzellent - sehr robust
- Error Rate 1-5%: Akzeptabel - Monitoring nötig
- Error Rate > 5%: Problematisch - dringende Action nötig

**Error-Kategorien:**
- **no_results**: Retrieval findet nichts → Datenqualität?
- **timeout**: System zu langsam → Performance-Problem
- **llm_error**: API-Fehler → Retry-Logic implementieren
- **parse_error**: Output-Format falsch → Prompt anpassen

**Wann nutzen?**
- ✅ **Production Monitoring** (kritisch!)
- ✅ Alerting & Incident Response
- ✅ System Stability Assessment

**Target:** < 5% Total Error Rate

---

## User Satisfaction Metrics

**Definition:** Production Feedback von echten Usern

**Metriken:**
```python
# python
# Thumbs Up/Down
thumbs_up_rate = thumbs_up / (thumbs_up + thumbs_down)
# Target: > 80%

# Follow-up Question Rate
followup_rate = queries_with_followup / total_queries
# Target: < 30% (niedrig = Antwort war gut)

# Session Abandonment
abandonment = sessions_abandoned / total_sessions
# Target: < 20%

# Average Session Length
avg_session_length = total_session_time / total_sessions
# Kein fixer Target (use-case abhängig)
```

**Interpretation:**

**Thumbs Up Rate:**
- > 80%: User sind sehr zufrieden
- 60-80%: Akzeptabel, aber Verbesserungspotential
- < 60%: Problematisch - User sind unzufrieden

**Follow-up Rate:**
- < 20%: Erste Antwort meist ausreichend (gut!)
- 20-40%: User brauchen oft Klarstellung
- > 40%: Antworten sind zu vage/unvollständig

**Abandonment:**
- < 10%: Exzellent - User bekommen was sie brauchen
- 10-20%: Normal - einige User geben auf
- > 20%: Problematisch - System hilft nicht genug

**Wann nutzen?**
- ✅ **Production** (wichtigste Business-Metrik!)
- ✅ A/B Testing
- ✅ Feature Priorisierung

**Target:**
- Thumbs Up Rate: > 80%
- Follow-up Rate: < 30%
- Abandonment: < 20%

---

## System Health Dashboard

**Kombinierte Übersicht:**

```python
# python
import pandas as pd

def system_health_report():
    metrics = {
        'Metric': [
            'Latency (P95)',
            'Throughput (QPS)',
            'Cost per Query',
            'Cache Hit Rate',
            'Error Rate',
            'Thumbs Up Rate'
        ],
        'Current': [
            f"{latency_p95:.2f}s",
            f"{qps:.1f}",
            f"${cost:.4f}",
            f"{hit_rate:.1%}",
            f"{error_rate:.1%}",
            f"{thumbs_up:.1%}"
        ],
        'Target': [
            "< 3s",
            "> 10",
            "< $0.05",
            "> 30%",
            "< 5%",
            "> 80%"
        ],
        'Status': [
            "✅" if latency_p95 < 3 else "❌",
            "✅" if qps > 10 else "❌",
            "✅" if cost < 0.05 else "❌",
            "✅" if hit_rate > 0.3 else "⚠️",
            "✅" if error_rate < 0.05 else "❌",
            "✅" if thumbs_up > 0.8 else "❌"
        ]
    }

    df = pd.DataFrame(metrics)
    return df

# Output
print(system_health_report().to_string(index=False))
```

**Output Beispiel:**
```
         Metric Current Target Status
   Latency (P95)   2.34s   < 3s     ✅
Throughput (QPS)    15.3   > 10     ✅
 Cost per Query  $0.034  < $0.05    ✅
 Cache Hit Rate   42.3%   > 30%     ✅
     Error Rate    2.1%    < 5%     ✅
Thumbs Up Rate   85.2%   > 80%     ✅
```

---

## Alerting & SLA

**SLA (Service Level Agreement) Beispiel:**

```python
# python
class SLAMonitor:
    def __init__(self):
        self.sla_config = {
            'latency_p95': {'threshold': 3.0, 'critical': True},
            'error_rate': {'threshold': 0.05, 'critical': True},
            'thumbs_up_rate': {'threshold': 0.75, 'critical': False},
        }

    def check_sla_breach(self, metrics):
        breaches = []

        if metrics['latency_p95'] > self.sla_config['latency_p95']['threshold']:
            breaches.append({
                'metric': 'latency_p95',
                'value': metrics['latency_p95'],
                'threshold': 3.0,
                'critical': True
            })

        if metrics['error_rate'] > self.sla_config['error_rate']['threshold']:
            breaches.append({
                'metric': 'error_rate',
                'value': metrics['error_rate'],
                'threshold': 0.05,
                'critical': True
            })

        return breaches

    def alert(self, breaches):
        for breach in breaches:
            severity = "CRITICAL" if breach['critical'] else "WARNING"
            print(f"{severity}: {breach['metric']} = {breach['value']:.3f} (threshold: {breach['threshold']})")
            # Send to Slack/PagerDuty/etc.
```

**Wann nutzen?**
- ✅ Production 24/7 Monitoring
- ✅ Incident Response
- ✅ SLA Compliance Tracking
