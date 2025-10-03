# Production & Deployment

Umfassender Guide für Production-Ready RAG-Systeme: Deployment, Monitoring, Scaling, Optimization.

---

## 📑 Inhaltsverzeichnis

1. [Deployment Strategies](#1-deployment-strategies) - Wie bringe ich das System live?
2. [API Design & Serving](#2-api-design--serving) - REST/gRPC Endpoints strukturieren
3. [Performance Optimization](#3-performance-optimization) - Latency, Throughput optimieren
4. [Scaling](#4-scaling) - Horizontal/Vertical Scaling, Load Balancing
5. [Monitoring & Observability](#5-monitoring--observability) - Metriken, Alerts, Debugging
6. [Caching Strategies](#6-caching-strategies) - Query Cache, Embedding Cache
7. [Security](#7-security) - Authentication, Rate Limiting, Input Validation
8. [Cost Optimization](#8-cost-optimization) - Kosten reduzieren ohne Quality-Loss

---

## 1. Deployment Strategies

### Container-basiert (Docker)

**Wann nutzen:**
- Standard für moderne Microservices
- Portabel zwischen Environments (Dev/Staging/Prod)
- Einfache Dependency-Verwaltung

**Vorteile:**
- ✅ Konsistente Environments
- ✅ Einfaches Rollback bei Problemen
- ✅ Resource Isolation

**Nachteile:**
- ❌ Overhead (Container Runtime)
- ❌ Image Size kann groß werden (Models!)

---

### Kubernetes (K8s)

**Wann nutzen:**
- Production mit hohen Availability-Anforderungen
- Auto-Scaling basierend auf Load
- Mehrere Services/Microservices

**Key Features:**
- **Replica Sets:** Automatisches Hoch/Runterskalieren
- **Health Checks:** Automatischer Neustart bei Failures
- **Load Balancing:** Traffic-Verteilung über Pods
- **Rolling Updates:** Zero-Downtime Deployments

**Deployment Workflow:**

```
1. Build Docker Image mit Model
2. Push zu Registry (Docker Hub, ECR, GCR)
3. Define K8s Deployment YAML (Replicas, Resources)
4. Apply Deployment → K8s startet Pods
5. Expose Service (LoadBalancer/Ingress)
```

**Resource Considerations:**

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| **Embedding Service** | 2-4 cores | 4-8GB | Model in RAM |
| **Vector Search** | 1-2 cores | 2-4GB + Index Size | FAISS/Weaviate |
| **LLM Generation** | 4-8 cores (or GPU) | 16-32GB | Wenn self-hosted |

---

### Serverless (AWS Lambda, Cloud Run)

**Wann nutzen:**
- Variable Load (nicht konstant)
- Cost-Sensitive (pay-per-use)
- Kein Ops-Team

**Vorteile:**
- ✅ Zero Ops: Keine Server-Verwaltung
- ✅ Auto-Scaling: Automatisch bei Traffic
- ✅ Pay-per-Request: Keine Idle-Kosten

**Nachteile:**
- ❌ **Cold Starts:** Erste Requests langsam (Model laden!)
- ❌ **Memory Limits:** AWS Lambda max 10GB
- ❌ **Timeout:** Lambda max 15min

**Workarounds für Cold Starts:**
- Warme Instances mit Provisioned Concurrency
- Kleinere Models (Distilled)
- Model-Lazy-Loading optimieren

---

## 2. API Design & Serving

### REST API Design

**Standard Endpoints:**

| Endpoint | Method | Input | Output | Use Case |
|----------|--------|-------|--------|----------|
| `/embed` | POST | `{texts: List[str]}` | `{embeddings: List[List[float]]}` | Batch Embedding |
| `/search` | POST | `{query: str, k: int}` | `{results: List[Doc]}` | Semantic Search |
| `/health` | GET | - | `{status: "healthy"}` | Health Check |
| `/metrics` | GET | - | Prometheus Metrics | Monitoring |

**Best Practices:**

1. **Batching:** Akzeptiere Arrays für besseren Throughput
2. **Pagination:** Bei `/search` Limit + Offset für große Result-Sets
3. **Versioning:** `/v1/embed`, `/v2/embed` für Breaking Changes
4. **Idempotency:** POST mit `request_id` für Retry-Safety

---

### Performance-Optimierung

#### 1. Model Loading

**Problem:** Model laden dauert Sekunden → Cold Start

**Lösung:**
- Model beim Service-Start laden (nicht per Request)
- Lazy Loading nur wenn Memory-kritisch
- Model Caching in shared Memory (multi-worker)

---

#### 2. Batching

**Problem:** Einzelne Requests ineffizient (GPU nicht ausgelastet)

**Lösung:**
- **Dynamic Batching:** Sammle Requests für kurze Zeit (10-50ms)
- **Batch Size:** 16-64 für optimale GPU-Auslastung

**Performance-Gain:**

| Batch Size | Throughput | Latency |
|-----------|-----------|---------|
| 1 | 10 req/s | 100ms |
| 16 | 80 req/s | 200ms |
| 64 | 200 req/s | 320ms |

**Trade-off:** Höhere Latency pro Request, aber viel mehr Throughput

---

#### 3. Quantization

**Embeddings komprimieren:**

| Precision | Memory | Performance-Loss | Speedup |
|-----------|--------|------------------|---------|
| **Float32** | Baseline | 0% | 1× |
| **Float16** | 50% | <1% | 1.5× |
| **INT8** | 75% | 1-3% | 2-3× |
| **Binary** | 96% | 5-15% | 10-30× |

**Empfehlung:**
- **INT8** für Production (best balance)
- **Binary** nur bei >10M Dokumenten

**Mehr Details:** [RAG-MODEL-ANALYSIS.md - Quantization](#)

---

## 3. Scaling

### Horizontal Scaling

**Konzept:** Mehr Instanzen statt größere Instanzen

**Wann:**
- Traffic steigt (>100 req/s)
- High Availability nötig (Redundanz)

**Setup:**
```
Load Balancer (Nginx, AWS ALB)
    ├─ Embedding Service Instance 1
    ├─ Embedding Service Instance 2
    └─ Embedding Service Instance 3
```

**Load Balancing Strategien:**

| Strategie | Wann nutzen | Vorteile |
|-----------|-------------|----------|
| **Round Robin** | Equal Capacity Instances | Einfach, fair |
| **Least Connections** | Variable Request-Dauer | Bessere Auslastung |
| **IP Hash** | Stateful Services | Consistent Routing |

---

### Vertical Scaling

**Konzept:** Größere Instance (mehr CPU/RAM/GPU)

**Wann:**
- Einfacher Setup (keine Distribution)
- GPU-bound Workload
- Weniger Koordinations-Overhead

**Limits:**
- Max Instance Size (z.B. AWS p4d.24xlarge)
- Kosten steigen nicht-linear

---

### Distributed Embedding

**Für sehr große Korpora (1M+ Dokumente):**

**Ansätze:**

| Methode | Tool | Complexity | Use Case |
|---------|------|-----------|----------|
| **Multi-GPU** | PyTorch DataParallel | ⭐⭐ | 2-8 GPUs, ein Server |
| **Multi-Node** | Ray, Dask | ⭐⭐⭐ | 10+ Nodes |
| **Simple Batching** | Python multiprocessing | ⭐ | CPU/Single GPU |

**Performance:** Linear Scaling (2 GPUs = 2× schneller)

---

## 4. Monitoring & Observability

### Key Metriken

#### Application Metriken

| Metrik | Was tracken | Alert-Threshold |
|--------|-------------|-----------------|
| **Request Latency (p50, p95, p99)** | Embedding/Search Zeit | p99 > 500ms |
| **Throughput** | Requests/Second | Drop >20% |
| **Error Rate** | 4xx, 5xx Errors | >1% |
| **Cache Hit Rate** | % Cache Hits | <80% |

#### System Metriken

| Metrik | Was tracken | Alert-Threshold |
|--------|-------------|-----------------|
| **CPU Usage** | % Auslastung | >80% sustained |
| **Memory Usage** | RAM + Swap | >85% |
| **GPU Utilization** | GPU % (falls GPU) | <30% (underutilized) |
| **Disk I/O** | Read/Write Ops | High I/O wait |

#### Model-spezifische Metriken

| Metrik | Was tracken | Warum wichtig |
|--------|-------------|---------------|
| **Embedding Dimensionality** | Output Shape | Detect Model-Änderungen |
| **Average Similarity Scores** | Mean Cosine Sim | Model Drift Detection |
| **Query Distribution** | Query Length, Tokens | Input Pattern Changes |

---

### Monitoring Tools

| Tool | Use Case | Complexity |
|------|----------|-----------|
| **Prometheus + Grafana** | Metriken sammeln + visualisieren | ⭐⭐⭐ |
| **DataDog** | All-in-one SaaS | ⭐⭐ |
| **CloudWatch** (AWS) | AWS-native | ⭐⭐ |
| **Simple Logging** | JSON Logs → Elasticsearch | ⭐ |

---

### Alerting

**Kritische Alerts:**

```
1. Error Rate >5% für 5min → PagerDuty
2. p99 Latency >1000ms für 10min → Slack
3. Service Down (Health Check fail) → PagerDuty
4. Memory >90% für 5min → Email
```

---

## 5. Caching Strategies

### Query Cache

**Konzept:** Gleiche Query → cached Response (ohne Recompute)

**Implementierung:**

| Cache-Typ | Tool | TTL | Use Case |
|-----------|------|-----|----------|
| **In-Memory** | Python dict, Redis | 1-24h | Häufige Queries |
| **Distributed** | Redis, Memcached | 1-7d | Multi-Instance Setup |
| **CDN** | CloudFront, Cloudflare | 24h | Public Search APIs |

**Cache Hit Rate:**
- **Target:** >80% für typische E-Commerce/Docs
- **Monitoring:** Cache Hits / Total Requests

**Cache Invalidation:**
- **Time-based:** TTL (Time-To-Live)
- **Event-based:** Bei Document-Update → Cache Clear

---

### Embedding Cache

**Konzept:** Dokumente ändern sich selten → Embeddings cachen

**Storage:**

| Option | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Vector DB Built-in** | Einfach | DB-abhängig | Standard |
| **Separate Cache** | Flexibel | Mehr Complexity | Custom Setup |
| **Persistent Disk** | Kein Re-Embedding | Langsam vs RAM | Large Corpus |

**Cache Size Berechnung:**
```
Embeddings Memory = Num_Docs × Dimensions × 4 bytes (float32)
Beispiel: 100k docs × 768 dims × 4 = ~300MB
```

**Invalidation-Strategie:**
- Document Update → Re-Embed + Update Vector DB
- Model Change → Full Re-Embedding (Background Job)

---

## 6. Security

### Authentication & Authorization

**API Keys:**
- **Basis:** Einfachste Methode
- **Rotation:** Keys regelmäßig rotieren
- **Scope:** Unterschiedliche Keys für Read/Write

**OAuth 2.0:**
- **Enterprise:** Für komplexe Permissions
- **Token-based:** JWT für stateless Auth

---

### Rate Limiting

**Warum:**
- DDoS Protection
- Fair Usage (Multi-Tenant)
- Cost Control

**Strategies:**

| Strategie | Limit | Use Case |
|-----------|-------|----------|
| **IP-based** | 100 req/min per IP | Public API |
| **API Key-based** | 1000 req/min per Key | Authenticated Users |
| **Token Bucket** | Burst erlaubt, avg limit | Flexible Limits |

**Implementierung:**
- **Nginx:** `limit_req_zone`
- **Redis:** Token bucket algorithm
- **Cloud:** AWS API Gateway Rate Limiting

---

### Input Validation

**Validierung:**
- **Query Length:** Max 500 chars (prevent abuse)
- **Batch Size:** Max 100 texts (prevent OOM)
- **Sanitization:** Remove/Escape special chars

**Beispiel-Validierung:**
```
- Query leer → 400 Bad Request
- Query >500 chars → 413 Payload Too Large
- Batch >100 → 413 Payload Too Large
- Invalid UTF-8 → 400 Bad Request
```

---

## 7. Cost Optimization

### Compute Costs

**Strategien:**

| Strategie | Einsparung | Trade-off |
|-----------|-----------|-----------|
| **Kleineres Model** | 50-70% | -5-10% Quality |
| **Distillation** | 60-80% | -5-15% Quality |
| **CPU statt GPU** | 70-90% | 5-10× langsamer |
| **Spot Instances** | 60-80% | Kann terminiert werden |

---

### Storage Costs

**Vector Index Compression:**

| Method | Size Reduction | Performance Impact |
|--------|----------------|-------------------|
| **INT8 Quantization** | 75% | Minimal (<3%) |
| **Binary Quantization** | 96% | Moderate (5-15%) |
| **PQ (Product Quantization)** | 90-95% | Low (2-8%) |

**Cost Beispiel:**
- 1M Docs × 768 dims × 4 bytes = 3GB (Float32)
- INT8: 750MB (-75%)
- Binary: 96MB (-96%)

---

### Request Costs

**Für Cloud APIs (OpenAI Embeddings, etc.):**

| Optimierung | Einsparung | Implementierung |
|-------------|-----------|-----------------|
| **Caching** | 70-90% | Redis Cache |
| **Deduplication** | 10-30% | Hash-basiert |
| **Batch Requests** | 20-40% | Bulk API nutzen |

---

## 8. Checklist: Production Readiness

### Before Go-Live

**Infrastructure:**
- [ ] Health Checks implementiert
- [ ] Auto-Scaling konfiguriert
- [ ] Load Balancer eingerichtet
- [ ] Backup-Strategie definiert

**Monitoring:**
- [ ] Metriken-Dashboard (Grafana/DataDog)
- [ ] Alerts konfiguriert (PagerDuty/Slack)
- [ ] Log-Aggregation (ELK/CloudWatch)
- [ ] Error Tracking (Sentry)

**Performance:**
- [ ] Load Testing durchgeführt (100x expected traffic)
- [ ] Latency <300ms (p95)
- [ ] Cache Hit Rate >80%
- [ ] Model Quantization angewendet

**Security:**
- [ ] Authentication implementiert
- [ ] Rate Limiting aktiv
- [ ] Input Validation
- [ ] HTTPS/TLS

**Cost:**
- [ ] Budget-Alerts eingerichtet
- [ ] Cost-Monitoring Dashboard
- [ ] Auto-Scaling Limits gesetzt

---

## Weitere Ressourcen

- [RAG-MODEL-ANALYSIS.md](RAG-MODEL-ANALYSIS.md) - Model Selection & Fine-Tuning
- [RAG-EVALUATION-GUIDE.md](RAG-EVALUATION-GUIDE.md) - Testing & Evaluation
- [RAG-EMBEDDING-STRATEGIES.md](RAG-EMBEDDING-STRATEGIES.md) - Embedding Basics
