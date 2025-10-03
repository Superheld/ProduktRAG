# Specialized Text Models - Complete Deep Dive

**Zweck:** Spezialisierte Text-Models für spezifische NLP-Tasks
**Scope:** NER, Classification, Translation, Summarization, Question Answering, etc.
**Zielgruppe:** Entwickler die über RAG hinaus NLP-Tasks verstehen und implementieren wollen

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Named Entity Recognition (NER)](#2-named-entity-recognition-ner)
3. [Text Classification](#3-text-classification)
4. [Translation](#4-translation)
5. [Summarization](#5-summarization)
6. [Question Answering](#6-question-answering)
7. [Sentiment Analysis](#7-sentiment-analysis)
8. [Text Generation (non-LLM)](#8-text-generation-non-llm)
9. [Information Extraction](#9-information-extraction)
10. [Production & Integration](#10-production--integration)

---

## 1. Overview

### 1.1 Warum Specialized Models?

**vs LLMs:**
```
LLM (GPT-4):
- Kann alles (generalistisch)
- Groß, langsam, teuer
- Braucht Prompts
- 8B-1.7T Parameter

Specialized Model:
- Kann eine Sache sehr gut
- Klein, schnell, günstig
- Fine-tuned für Task
- 100M-400M Parameter
```

**Wann Specialized Models?**
- ✅ Task ist klar definiert (NER, Classification, etc.)
- ✅ Viele Requests (Kosten sparen)
- ✅ Latency wichtig (<100ms)
- ✅ Offline/Edge Deployment
- ✅ Datenschutz (lokal statt API)

**Wann LLMs?**
- ✅ Flexibilität wichtig (Task ändert sich)
- ✅ Few-Shot Learning (wenig Trainingsdaten)
- ✅ Komplexe Reasoning-Aufgaben
- ✅ Mehrere Tasks gleichzeitig

---

### 1.2 Task Categories

**Token-Level Tasks:**
- Named Entity Recognition (NER)
- Part-of-Speech Tagging (POS)
- Token Classification

**Sequence-Level Tasks:**
- Text Classification
- Sentiment Analysis
- Intent Detection

**Sequence-to-Sequence Tasks:**
- Translation
- Summarization
- Paraphrasing

**Span-Level Tasks:**
- Question Answering (Extractive)
- Relation Extraction

---

## 2. Named Entity Recognition (NER)

### 2.1 Was ist NER?

**Definition:**
> Identifiziere und klassifiziere Entities (Personen, Orte, Organisationen, etc.) in Text

**Beispiel:**
```
Input: "Der LABO-288 von Kirsch wird in München hergestellt."

Output:
- "LABO-288": PRODUCT
- "Kirsch":   ORGANIZATION
- "München":  LOCATION
```

**Standard Entity Types:**
- **PER** - Person (Angela Merkel)
- **LOC** - Location (Berlin)
- **ORG** - Organization (Siemens)
- **MISC** - Miscellaneous (Euro, Olympics)

**Domain-Specific (z.B. Medizin):**
- **DISEASE** - Krankheit
- **DRUG** - Medikament
- **SYMPTOM** - Symptom
- **TREATMENT** - Behandlung

---

### 2.2 Tagging Schemes

**BIO (Begin, Inside, Outside):**
```
Text:   "Angela Merkel lebt in Berlin"
Tags:   B-PER  I-PER   O     O  B-LOC

B-PER: Begin Person
I-PER: Inside Person (Fortsetzung)
O:     Outside (kein Entity)
```

**BILOU (Begin, Inside, Last, Outside, Unit):**
```
Text:   "Angela Merkel lebt in Berlin"
Tags:   B-PER  L-PER   O     O  U-LOC

L-PER: Last token of Person
U-LOC: Unit (single-token entity)
```

**Vorteil BILOU:**
- ✅ Besser bei Ein-Token-Entities
- ✅ Explizites Ende-Signal

---

### 2.3 Models

#### **spaCy**

**Was ist spaCy?**
> Industrial-Strength NLP Library mit vortrainierten Pipelines

**Installation:**
```bash
pip install spacy

# Download German Model
python -m spacy download de_core_news_lg

# Download English Model
python -m spacy download en_core_web_trf  # Transformer-based
```

**Models:**
```
German:
- de_core_news_sm:  Small (13MB, CPU-friendly)
- de_core_news_md:  Medium (40MB)
- de_core_news_lg:  Large (500MB, beste Accuracy)

English:
- en_core_web_sm:   Small
- en_core_web_trf:  Transformer-based (RoBERTa)
```

**Usage:**
```python
import spacy

# Load Model
nlp = spacy.load("de_core_news_lg")

# Process Text
doc = nlp("Der LABO-288 von Kirsch wird in München hergestellt.")

# Extract Entities
for ent in doc.ents:
    print(f"{ent.text:15} → {ent.label_:10} (Confidence: {ent._.score:.2f})")

# Output:
# LABO-288        → MISC       (Confidence: 0.85)
# Kirsch          → ORG        (Confidence: 0.92)
# München         → LOC        (Confidence: 0.98)
```

**Entity Linking:**
```python
# Link zu Knowledge Base (Wikipedia, etc.)
for ent in doc.ents:
    if ent.kb_id_:
        print(f"{ent.text} → {ent.kb_id_}")
        # München → Q1726 (Wikidata ID)
```

**Custom NER:**
```python
from spacy.training import Example

# Training Data
TRAIN_DATA = [
    ("Der LABO-288 ist ein Kühlschrank", {
        "entities": [(4, 12, "PRODUCT")]
    }),
    ("Kirsch produziert Laborkühlschränke", {
        "entities": [(0, 6, "MANUFACTURER")]
    })
]

# Train
nlp = spacy.blank("de")
ner = nlp.add_pipe("ner")
ner.add_label("PRODUCT")
ner.add_label("MANUFACTURER")

# Training Loop
for epoch in range(10):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example])

# Save
nlp.to_disk("./custom_ner_model")
```

**Vorteile:**
- ✅ Production-ready
- ✅ Sehr schnell (optimierter Code)
- ✅ Viele Sprachen
- ✅ Einfache API

**Nachteile:**
- ❌ Transformer-Models langsamer als BERT-native
- ❌ Custom NER braucht viele Daten (1000+ Beispiele)

---

#### **Hugging Face Transformers (BERT-based)**

**Models:**
```
German:
- dbmdz/bert-large-cased-finetuned-conll03-german
- deepset/gbert-large (für Fine-Tuning)

English:
- dslim/bert-base-NER
- Jean-Baptiste/roberta-large-ner-english

Multilingual:
- Davlan/xlm-roberta-base-ner-hrl
```

**Usage:**
```python
from transformers import pipeline

# Load NER Pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-german")

# Predict
text = "Angela Merkel wohnt in Berlin"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:15} → {entity['entity']:10} (Score: {entity['score']:.2f})")

# Output:
# Angela          → B-PER       (Score: 0.99)
# Merkel          → I-PER       (Score: 0.99)
# Berlin          → B-LOC       (Score: 0.99)
```

**Aggregation (Tokens zu Entities):**
```python
# Problem: BERT tokenisiert Wörter (Angela → [Angela])
# → WordPiece Tokens müssen zusammengefügt werden

ner = pipeline("ner", model="...", aggregation_strategy="simple")

entities = ner(text)
for entity in entities:
    print(f"{entity['word']:15} → {entity['entity_group']:10}")

# Output:
# Angela Merkel   → PER
# Berlin          → LOC
```

**Fine-Tuning:**
```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load Dataset (CoNLL-2003 format)
dataset = load_dataset("conll2003")

# Model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-german-cased",
    num_labels=9  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
)

# Training
training_args = TrainingArguments(
    output_dir="./ner_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

trainer.train()
```

**Vorteile:**
- ✅ State-of-the-art Accuracy
- ✅ Viele vortrainierte Models
- ✅ Einfaches Fine-Tuning

**Nachteile:**
- ❌ Langsamer als spaCy (BERT inference)
- ❌ Mehr VRAM

---

#### **Flair**

**Was ist Flair?**
> Character-level NLP Framework (sehr gut für seltene Words)

**Installation:**
```bash
pip install flair
```

**Usage:**
```python
from flair.data import Sentence
from flair.models import SequenceTagger

# Load Tagger
tagger = SequenceTagger.load("de-ner-large")

# Predict
sentence = Sentence("Angela Merkel wohnt in Berlin")
tagger.predict(sentence)

# Extract
for entity in sentence.get_spans('ner'):
    print(f"{entity.text:15} → {entity.tag:10} (Confidence: {entity.score:.2f})")
```

**Stacking (mehrere Embeddings kombinieren):**
```python
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# Character-level + Word-level
embeddings = StackedEmbeddings([
    WordEmbeddings('de'),      # Word2Vec
    FlairEmbeddings('de-forward'),  # Character-level (forward)
    FlairEmbeddings('de-backward')  # Character-level (backward)
])

# → Sehr robust für Typos, seltene Wörter
```

**Vorteile:**
- ✅ Character-level → gut für Typos, Neologismen
- ✅ Sehr gute Accuracy
- ✅ Einfaches Stacking

**Nachteile:**
- ❌ Langsam (Character-level RNN)
- ❌ Mehr Memory

---

### 2.4 Evaluation Metrics

**Precision, Recall, F1:**
```
Gold:      "Angela Merkel wohnt in Berlin"
           B-PER  I-PER   O     O  B-LOC

Predicted: "Angela Merkel wohnt in Berlin"
           B-PER  I-PER   O     B-LOC B-LOC  ← Fehler!

True Positives (TP):  Angela Merkel (PER), Berlin (LOC) = 2
False Positives (FP): in (LOC) = 1
False Negatives (FN): 0

Precision = TP / (TP + FP) = 2 / 3 = 0.67
Recall    = TP / (TP + FN) = 2 / 2 = 1.0
F1        = 2 × (P × R) / (P + R) = 0.80
```

**Exact Match vs Partial Match:**
```
Exact Match:  Entity-Grenzen müssen exakt stimmen
Partial Match: Überlappung zählt

Gold:      "Angela Merkel"
Predicted: "Merkel"

Exact:   0 TP (Grenzen falsch)
Partial: 1 TP (Überlappung vorhanden)
```

**Strict vs Lenient:**
```
Strict:  Entity-Text UND Entity-Type müssen stimmen
Lenient: Nur Entity-Text muss stimmen

Gold:      "München" → LOC
Predicted: "München" → ORG

Strict:   0 TP (falscher Type)
Lenient:  1 TP (richtiger Text)
```

---

### 2.5 NER für RAG

**Use Cases:**

**1. Query Understanding**
```python
query = "Zeige mir Kühlschränke von Liebherr in München"

# NER
entities = ner(query)
# → Liebherr: MANUFACTURER
# → München:  LOCATION

# Structured Query
structured_query = {
    "manufacturer": "Liebherr",
    "location": "München"
}

# Combine: Semantic Search + Filters
results = vector_db.query(
    query_embedding,
    filter={"manufacturer": "Liebherr", "location": "München"}
)
```

**2. Entity Linking in Context**
```python
# Retrieved Chunks enthalten Entities
chunk = "Der LABO-288 von Kirsch..."

# Extract Entities
entities = ner(chunk)

# Metadata Enrichment
chunk_metadata = {
    "products": ["LABO-288"],
    "manufacturers": ["Kirsch"],
    "entities": entities
}

# Nutze für Filtering/Ranking
```

**3. Anonymisierung**
```python
# Remove PII (Personal Identifiable Information)
text = "Kunde Herr Müller aus Berlin hat angerufen"

entities = ner(text)
for entity in entities:
    if entity['entity_group'] in ['PER', 'LOC']:
        text = text.replace(entity['word'], f"[{entity['entity_group']}]")

# → "Kunde [PER] aus [LOC] hat angerufen"
```

---

## 3. Text Classification

### 3.1 Was ist Text Classification?

**Definition:**
> Ordne Text einer oder mehreren Kategorien zu

**Typen:**

**Binary Classification:**
```
Spam Detection:  "Spam" vs "Not Spam"
Sentiment:       "Positive" vs "Negative"
```

**Multi-Class:**
```
Topic:  "Sports", "Politics", "Technology", "Entertainment"
Intent: "Question", "Command", "Statement"
```

**Multi-Label:**
```
Tags: ["Python", "Machine Learning", "Tutorial"]
      → Ein Text kann mehrere Labels haben
```

---

### 3.2 Models

#### **DistilBERT (schnell)**

**Was ist DistilBERT?**
> Kleinere, schnellere Version von BERT (via Knowledge Distillation)

**Stats:**
```
BERT:       110M Parameter, 100% Accuracy
DistilBERT: 66M Parameter,  97% Accuracy, 60% schneller ✅
```

**Usage:**
```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("This product is amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Custom Classification
classifier = pipeline("text-classification", model="your-fine-tuned-model")
result = classifier("Der LABO-288 ist ein Laborkühlschrank")
print(result)
# [{'label': 'PRODUCT_DESCRIPTION', 'score': 0.95}]
```

**Fine-Tuning:**
```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Dataset
dataset = load_dataset("imdb")  # Movie Reviews

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Positive, Negative
)

# Training
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()
```

**Vorteile:**
- ✅ Schnell (60% faster als BERT)
- ✅ Klein (66M Parameter)
- ✅ Gute Accuracy (97% von BERT)

**Nachteile:**
- ❌ Etwas schlechter als Full BERT

---

#### **RoBERTa (robust)**

**Was ist RoBERTa?**
> Robustly Optimized BERT - Verbessertes BERT Training

**Improvements:**
- Mehr Daten (160GB vs 16GB)
- Längeres Training
- Größere Batches
- Kein Next-Sentence-Prediction (war nicht hilfreich)

**Usage:**
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

result = classifier("I love this product!")
# [{'label': 'LABEL_2', 'score': 0.98}]  # LABEL_2 = Positive
```

**Vorteile:**
- ✅ Bessere Accuracy als BERT
- ✅ Robuster

**Nachteile:**
- ❌ Größer/langsamer als DistilBERT

---

#### **DeBERTa (state-of-the-art)**

**Was ist DeBERTa?**
> Decoding-enhanced BERT - Microsoft's BERT-Verbesserung

**Key Innovation:**
- Disentangled Attention (Content + Position separat)
- Enhanced Mask Decoder

**Performance:**
```
GLUE Benchmark (Higher = Better):
BERT:       80.5
RoBERTa:    88.5
DeBERTa:    90.3 ✅ State-of-the-art
```

**Usage:**
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="microsoft/deberta-v3-base")

# Fine-tune für deine Task
```

**Vorteile:**
- ✅ State-of-the-art Accuracy

**Nachteile:**
- ❌ Langsamer als BERT/RoBERTa
- ❌ Größer

---

### 3.3 Zero-Shot Classification

**Konzept:**
> Klassifiziere ohne Training-Daten für die spezifische Task

**Model:**
```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "Der LABO-288 ist ein Laborkühlschrank mit 280 Litern"

# Candidate Labels (beliebig!)
labels = ["Produktbeschreibung", "Technische Specs", "Preisinfo", "Kundenbewertung"]

result = classifier(text, labels)

print(result)
# {
#   'sequence': '...',
#   'labels': ['Produktbeschreibung', 'Technische Specs', 'Kundenbewertung', 'Preisinfo'],
#   'scores': [0.95, 0.78, 0.12, 0.08]
# }
```

**Multilingual:**
```python
# German Zero-Shot
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

labels = ["Produktbeschreibung", "Technische Daten", "Preis"]
result = classifier(text, labels)
```

**Vorteile:**
- ✅ Kein Training nötig
- ✅ Flexibel (Labels änderbar)
- ✅ Gut für Exploration

**Nachteile:**
- ❌ Schlechter als Fine-Tuned Model
- ❌ Langsam (NLI-Modell im Hintergrund)

---

### 3.4 Classification für RAG

**Use Cases:**

**1. Query Intent Detection**
```python
query = "Was kostet der LABO-288?"

intent = classifier(query, labels=["question", "command", "statement"])
# → "question"

# Route basierend auf Intent
if intent == "question":
    # RAG Pipeline
    response = rag_pipeline(query)
elif intent == "command":
    # Action (z.B. "Bestelle LABO-288")
    response = execute_command(query)
```

**2. Chunk Classification**
```python
# Klassifiziere Chunks für besseres Retrieval
chunk = "Der LABO-288 kostet 2.500€"

chunk_type = classifier(chunk, labels=["description", "specs", "price", "review"])
# → "price"

# Metadata
chunk_metadata = {
    "type": "price",
    "content": chunk
}

# Query-specific Retrieval
# Wenn Query nach Preis fragt → nur Price-Chunks
```

**3. Content Moderation**
```python
user_query = "..."

is_safe = classifier(user_query, labels=["safe", "unsafe"])

if is_safe['label'] == 'unsafe':
    return "Sorry, I can't help with that."
```

---

## 4. Translation

### 4.1 Models

#### **MarianMT (Helsinki-NLP)**

**Was ist MarianMT?**
> Open-Source Translation Models für viele Sprachpaare

**Available Pairs:**
```
opus-mt-de-en  (German → English)
opus-mt-en-de  (English → German)
opus-mt-multi-en (viele Sprachen → English)
```

**Usage:**
```python
from transformers import pipeline

# German → English
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

result = translator("Der LABO-288 ist ein Laborkühlschrank")
print(result[0]['translation_text'])
# → "The LABO-288 is a laboratory refrigerator"

# English → German
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

result = translator("The product is very good")
print(result[0]['translation_text'])
# → "Das Produkt ist sehr gut"
```

**Batching:**
```python
texts = [
    "Hallo Welt",
    "Wie geht es dir?",
    "Ich mag Kühlschränke"
]

results = translator(texts)
for result in results:
    print(result['translation_text'])
```

**Vorteile:**
- ✅ Viele Sprachpaare (1000+)
- ✅ Open Source
- ✅ Lokal ausführbar

**Nachteile:**
- ❌ Schlechter als kommerzielle APIs (Google, DeepL)
- ❌ Domänenspezifische Fehler

---

#### **NLLB (No Language Left Behind, Meta)**

**Was ist NLLB?**
> Multilingual Translation für 200 Sprachen

**Models:**
```
nllb-200-600M:    600M Parameter (schnell)
nllb-200-1.3B:    1.3B Parameter
nllb-200-3.3B:    3.3B Parameter (beste Quality)
```

**Usage:**
```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="facebook/nllb-200-600M",
    src_lang="deu_Latn",  # German
    tgt_lang="eng_Latn"   # English
)

result = translator("Der LABO-288 ist ein Kühlschrank")
print(result[0]['translation_text'])
```

**Language Codes:**
```
deu_Latn: German
eng_Latn: English
fra_Latn: French
spa_Latn: Spanish
...
```

**Vorteile:**
- ✅ 200 Sprachen
- ✅ Gut für low-resource Languages
- ✅ Open Source

**Nachteile:**
- ❌ Groß (600M-3.3B)
- ❌ Langsamer als MarianMT

---

#### **M2M-100 (Meta)**

**Was ist M2M-100?**
> Many-to-Many Translation (direkt ohne English-Pivot)

**Innovation:**
```
Traditional: DE → EN → FR (2 Steps)
M2M:         DE → FR (1 Step, direkter)
```

**Usage:**
```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Set source language
tokenizer.src_lang = "de"

# Encode
encoded = tokenizer("Der LABO-288 ist ein Kühlschrank", return_tensors="pt")

# Generate (target language: French)
generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("fr"))

# Decode
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(translation)
# → "Le LABO-288 est un réfrigérateur"
```

**Vorteile:**
- ✅ Direkte Translation (kein Pivot)
- ✅ 100 Sprachen

**Nachteile:**
- ❌ Komplexere API
- ❌ Groß (418M-1.2B)

---

#### **Commercial APIs**

**DeepL (beste Quality):**
```python
import deepl

translator = deepl.Translator("YOUR_API_KEY")

result = translator.translate_text(
    "Der LABO-288 ist ein Laborkühlschrank",
    target_lang="EN-US"
)

print(result.text)
# → "The LABO-288 is a laboratory refrigerator"
```

**Kosten:**
```
DeepL Free:  500k Zeichen/Monat
DeepL Pro:   €5.49 / Monat für 1M Zeichen
```

**Google Translate:**
```python
from google.cloud import translate_v2 as translate

translator = translate.Client()

result = translator.translate(
    "Der LABO-288 ist ein Kühlschrank",
    target_language="en"
)

print(result['translatedText'])
```

**Kosten:**
```
$20 / 1M Zeichen
```

**Quality Comparison:**
```
DeepL:       ⭐⭐⭐⭐⭐ (beste, besonders DE↔EN)
Google:      ⭐⭐⭐⭐
NLLB:        ⭐⭐⭐
MarianMT:    ⭐⭐⭐
```

---

### 4.2 Translation für RAG

**Use Cases:**

**1. Multilingual RAG**
```python
# User Query in German
query_de = "Was ist ein Laborkühlschrank?"

# Translate to English (if knowledge base is English)
query_en = translator(query_de, src_lang="de", tgt_lang="en")
# → "What is a laboratory refrigerator?"

# Retrieve (English)
chunks_en = vector_db.query(query_en)

# Generate Response (English)
response_en = llm.generate(chunks_en, query_en)

# Translate back to German
response_de = translator(response_en, src_lang="en", tgt_lang="de")
```

**2. Cross-Lingual Search**
```python
# Embed Query in multiple languages
query_de = "Laborkühlschrank"
query_en = translator(query_de)  # "Laboratory refrigerator"

emb_de = embedding_model.encode(query_de)
emb_en = embedding_model.encode(query_en)

# Average embeddings
emb_combined = (emb_de + emb_en) / 2

# Search
results = vector_db.query(emb_combined)
```

---

## 5. Summarization

### 5.1 Extractive vs Abstractive

**Extractive Summarization:**
> Wähle wichtigste Sätze aus Original-Text

```
Original: "Der LABO-288 ist ein Laborkühlschrank. Er hat 280 Liter Volumen. Die Temperatur liegt zwischen 0-15°C. Das Gerät hat eine Alarmfunktion."

Extractive: "Der LABO-288 ist ein Laborkühlschrank. Er hat 280 Liter Volumen."

→ Sätze 1 & 2 extrahiert
```

**Abstractive Summarization:**
> Generiere neue Zusammenfassung

```
Abstractive: "Der LABO-288 ist ein 280L Laborkühlschrank mit Temperaturbereich 0-15°C und Alarm."

→ Neu formuliert
```

---

### 5.2 Models

#### **BART (Facebook)**

**Was ist BART?**
> Denoising Autoencoder for Sequence-to-Sequence

**Architecture:**
- Encoder-Decoder (wie T5)
- Pre-training: Text korruptieren → rekonstruieren
- Good for: Summarization, Translation

**Usage:**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Der LABO-288 ist ein Laborkühlschrank von Kirsch. Das Gerät verfügt über ein Nutzvolumen von 280 Litern und bietet einen Temperaturbereich von 0 bis 15°C. Die statische Belüftung sorgt für eine gleichmäßige Temperaturverteilung. Eine elektronische Temperatursteuerung mit Digitalanzeige ermöglicht eine präzise Einstellung. Das Gerät ist mit einer Alarmfunktion ausgestattet, die bei Temperaturabweichungen oder offener Tür warnt. Die automatische Abtauung reduziert den Wartungsaufwand. Der LABO-288 entspricht der DIN 13221 Norm für Laborkühlschränke.
"""

summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
print(summary[0]['summary_text'])

# Output:
# "Der LABO-288 ist ein 280L Laborkühlschrank mit Temperaturbereich 0-15°C, elektronischer Steuerung und Alarmfunktion. Das Gerät entspricht DIN 13221."
```

**Parameters:**
```python
summarizer(
    text,
    max_length=130,      # Max Länge der Summary
    min_length=30,       # Min Länge
    do_sample=False,     # Greedy Decoding (deterministisch)
    num_beams=4,         # Beam Search (bessere Quality)
    length_penalty=2.0,  # Bevorzuge längere Summaries
    early_stopping=True
)
```

**Vorteile:**
- ✅ Sehr gute Abstractive Summaries
- ✅ Flexibel (Länge einstellbar)

**Nachteile:**
- ❌ Langsam (Seq2Seq)
- ❌ Englisch-fokussiert

---

#### **T5 (Google)**

**Was ist T5?**
> Text-to-Text Transfer Transformer - Alles ist Seq2Seq

**Konzept:**
```
Input:  "summarize: <TEXT>"
Output: "<SUMMARY>"

Input:  "translate English to German: Hello"
Output: "Hallo"

→ Unified Format für alle Tasks
```

**Usage:**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="t5-base")

text = "summarize: " + long_text  # Prefix!

summary = summarizer(text, max_length=100)
print(summary[0]['summary_text'])
```

**Vorteile:**
- ✅ Sehr flexibel (Multi-Task)
- ✅ Gute Quality

**Nachteile:**
- ❌ Prefix nötig ("summarize:")
- ❌ Englisch-fokussiert

---

#### **Pegasus (Google)**

**Was ist Pegasus?**
> Pre-training mit Gap Sentence Generation (speziell für Summarization)

**Pre-training:**
```
Original: "Sentence 1. Sentence 2. Sentence 3."
Masked:   "Sentence 1. [MASK]. Sentence 3."
Target:   "Sentence 2"

→ Model lernt wichtigste Sätze zu identifizieren
```

**Usage:**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

summary = summarizer(long_text, max_length=100)
```

**Vorteile:**
- ✅ State-of-the-art für Summarization
- ✅ Sehr gute Abstractive Summaries

**Nachteile:**
- ❌ Groß/langsam
- ❌ Englisch-fokussiert

---

#### **German Summarization**

**mT5 (Multilingual T5):**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="google/mt5-base")

# Deutsch funktioniert!
summary = summarizer(german_text, max_length=100)
```

**mBART:**
```python
summarizer = pipeline("summarization", model="facebook/mbart-large-cc25")

summary = summarizer(german_text)
```

**Vorteile:**
- ✅ Multilingual

**Nachteile:**
- ❌ Schlechter als spezialisierte deutsche Models
- ❌ Weniger getestet

---

### 5.3 Summarization für RAG

**Use Cases:**

**1. Long Document Compression**
```python
# Problem: Chunk zu lang für Context Window
long_chunk = "..."  # 2000 Tokens

# Summarize
summary = summarizer(long_chunk, max_length=500)

# Use Summary statt Original
context = summary
```

**2. Multi-Document Summarization**
```python
# Top-10 Chunks retrieved
chunks = retrieve_top_k(query, k=10)

# Combine
combined = "\n\n".join(chunks)

# Summarize
summary = summarizer(combined, max_length=300)

# Use als Context
response = llm.generate(query, context=summary)
```

**3. Response Summarization**
```python
# LLM generiert lange Antwort
long_response = llm.generate(query, context)  # 1000 Tokens

# User will kurze Antwort
short_response = summarizer(long_response, max_length=100)
```

---

## 6. Question Answering

### 6.1 Extractive QA

**Konzept:**
> Finde Antwort-Span im gegebenen Kontext

**Example:**
```
Context: "Der LABO-288 hat ein Volumen von 280 Litern und kostet 2.500€."
Question: "Wie viel kostet der LABO-288?"

Answer: "2.500€" (extrahiert aus Context)
        ^^^^^^^^
        Start: 48, End: 54
```

---

#### **Models**

**BERT for QA:**
```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

context = "Der LABO-288 hat ein Volumen von 280 Litern und kostet 2.500€."
question = "Wie viel kostet der LABO-288?"

result = qa(question=question, context=context)

print(result)
# {
#   'answer': '2.500€',
#   'score': 0.95,
#   'start': 48,
#   'end': 54
# }
```

**German QA:**
```python
qa = pipeline("question-answering", model="deepset/gbert-base-germanquad")

result = qa(question=question, context=context)
```

**Long Context (Chunked):**
```python
# Problem: Context zu lang (>512 Tokens)

# Lösung 1: Chunking
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    handle_impossible_answer=True,  # "No answer" möglich
    max_seq_len=512,
    doc_stride=128  # Overlap between chunks
)

result = qa(question=long_question, context=very_long_context)
```

**Vorteile:**
- ✅ Schnell
- ✅ Exact Answer (Span)
- ✅ Confidence Score

**Nachteile:**
- ❌ Nur Extractive (keine Synthese)
- ❌ Braucht genauen Context
- ❌ Schlechter bei komplexen Fragen

---

### 6.2 Generative QA

**Konzept:**
> Generiere Antwort (nicht extrahieren)

**Models:**
- T5
- BART
- LLMs (GPT, Llama)

**Usage (T5):**
```python
from transformers import pipeline

qa = pipeline("text2text-generation", model="t5-base")

input_text = f"""
question: Wie viel kostet der LABO-288?
context: Der LABO-288 hat ein Volumen von 280 Litern und kostet 2.500€.
"""

result = qa(input_text, max_length=50)
print(result[0]['generated_text'])
# → "Der LABO-288 kostet 2.500€."
```

**Vorteile:**
- ✅ Flexibler (kann paraphrasieren)
- ✅ Multi-Hop Reasoning

**Nachteile:**
- ❌ Hallucination Risk
- ❌ Langsamer
- ❌ Schwerer zu evaluieren

---

### 6.3 QA für RAG

**Hybrid Approach:**
```python
# 1. Retrieve relevant chunks
chunks = retrieve_top_k(query, k=5)

# 2. Extractive QA auf jedem Chunk
qa = pipeline("question-answering")

answers = []
for chunk in chunks:
    try:
        answer = qa(question=query, context=chunk)
        answers.append(answer)
    except:
        pass  # No answer in chunk

# 3. Rank by score
answers = sorted(answers, key=lambda x: x['score'], reverse=True)

# 4. Return top answer
best_answer = answers[0]['answer']
```

---

## 7. Sentiment Analysis

### 7.1 Models

**Binary Sentiment:**
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

result = sentiment("This product is amazing!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

result = sentiment("This product is terrible!")
# [{'label': 'NEGATIVE', 'score': 0.9995}]
```

**Fine-Grained (1-5 Stars):**
```python
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

result = sentiment("Das Produkt ist okay")
# [{'label': '3 stars', 'score': 0.65}]
```

**German:**
```python
sentiment = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

result = sentiment("Dieses Produkt ist fantastisch!")
# [{'label': 'positive', 'score': 0.99}]
```

---

### 7.2 Aspect-Based Sentiment

**Konzept:**
> Sentiment pro Aspekt

**Example:**
```
Review: "Das Essen war gut, aber der Service war schlecht"

Aspect Sentiments:
- Essen:   Positive
- Service: Negative
```

**Implementation:**
```python
# 1. Extract Aspects (NER oder Keywords)
aspects = ["Essen", "Service"]

# 2. Sentiment pro Aspect
aspect_sentiments = {}
for aspect in aspects:
    # Filter sentences mentioning aspect
    relevant_text = extract_sentences_with(review, aspect)

    # Sentiment
    sentiment = sentiment_analyzer(relevant_text)
    aspect_sentiments[aspect] = sentiment

# Output:
# {
#   'Essen': {'label': 'POSITIVE', 'score': 0.92},
#   'Service': {'label': 'NEGATIVE', 'score': 0.88}
# }
```

---

### 7.3 Sentiment für RAG

**Use Cases:**

**1. Filter Reviews by Sentiment**
```python
query = "Was sind positive Aspekte vom LABO-288?"

# Retrieve Reviews
reviews = retrieve_reviews(product_id="LABO-288")

# Filter Positive
positive_reviews = []
for review in reviews:
    sentiment = sentiment_analyzer(review)
    if sentiment['label'] == 'POSITIVE':
        positive_reviews.append(review)

# Generate Response from Positive Reviews
response = llm.generate(query, context=positive_reviews)
```

**2. Sentiment-Aware Response**
```python
user_query = "Ich bin unzufrieden mit dem LABO-288"

sentiment = sentiment_analyzer(user_query)
# → NEGATIVE

# Adjust Response Tone
if sentiment['label'] == 'NEGATIVE':
    system_prompt = "Du bist ein empathischer Support-Agent. Der Kunde ist unzufrieden."
else:
    system_prompt = "Du bist ein hilfreicher Assistent."

response = llm.generate(user_query, system_prompt=system_prompt)
```

---

## 8. Text Generation (non-LLM)

### 8.1 GPT-2 (small-scale)

**Usage:**
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The LABO-288 is a laboratory refrigerator with"

result = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    temperature=0.8
)

for i, output in enumerate(result):
    print(f"{i+1}. {output['generated_text']}")
```

**German:**
```python
generator = pipeline("text-generation", model="dbmdz/german-gpt2")

result = generator("Der LABO-288 ist ein Laborkühlschrank mit")
```

**Vorteile:**
- ✅ Klein (124M-1.5B)
- ✅ Schnell
- ✅ Lokal

**Nachteile:**
- ❌ Schlechte Quality vs moderne LLMs
- ❌ Keine Instruktionsbefolgung

---

## 9. Information Extraction

### 9.1 Relation Extraction

**Konzept:**
> Extrahiere Relationen zwischen Entities

**Example:**
```
Text: "Der LABO-288 wird von Kirsch hergestellt"

Relations:
- (LABO-288, manufactured_by, Kirsch)
```

**Models:**
```python
# spaCy mit Custom Relation Extraction
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_trf")

# Custom Relation Patterns
patterns = [
    {"label": "MANUFACTURED_BY", "pattern": [
        {"ENT_TYPE": "PRODUCT"},
        {"LOWER": {"IN": ["von", "by", "from"]}},
        {"ENT_TYPE": "ORG"}
    ]}
]

# Extract
doc = nlp("Der LABO-288 wird von Kirsch hergestellt")

for ent in doc.ents:
    if ent.label_ == "PRODUCT":
        # Look for manufacturer
        for token in ent.root.head.children:
            if token.ent_type_ == "ORG":
                print(f"{ent.text} manufactured_by {token.text}")
```

---

### 9.2 Event Extraction

**Konzept:**
> Extrahiere Ereignisse

**Example:**
```
Text: "Kirsch kündigte am 15.03.2024 die Markteinführung des LABO-300 an"

Event:
- Type: Produkteinführung
- Agent: Kirsch
- Date: 15.03.2024
- Product: LABO-300
```

---

## 10. Production & Integration

### 10.1 Model Serving

**FastAPI Endpoint:**
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load Model on Startup
ner = None

@app.on_event("startup")
async def load_model():
    global ner
    ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-german")

@app.post("/ner")
async def extract_entities(text: str):
    entities = ner(text)
    return {"entities": entities}

# Run: uvicorn app:app --reload
```

**Request:**
```bash
curl -X POST "http://localhost:8000/ner" \
  -H "Content-Type: application/json" \
  -d '{"text": "Angela Merkel wohnt in Berlin"}'
```

---

### 10.2 Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_ner(text: str):
    return ner(text)

# Häufige Queries werden gecached
```

---

### 10.3 Batching

```python
# Batch Requests für Throughput
texts = [
    "Text 1",
    "Text 2",
    "Text 3"
]

# Statt:
# for text in texts: ner(text)  # Langsam

# Besser:
results = ner(texts)  # Batch (schneller!)
```

---

## Summary: Quick Guide

### Task Selection:
```
Entity Extraction:    NER (spaCy, BERT)
Text Classification:  DistilBERT, RoBERTa
Translation:          DeepL API (beste) oder MarianMT (lokal)
Summarization:        BART, Pegasus
QA (Extractive):      BERT for QA
Sentiment:            transformers sentiment pipeline
```

### When to use vs LLMs:
```
Use Specialized:  High volume, low latency, cost-sensitive
Use LLM:          Flexible, complex reasoning, few-shot learning
```

---

**Navigation:**
- [← Back: LLM Architectures](02-LLM-ARCHITECTURES.md)
- [→ Next: Vision Models](04-VISION-MODELS.md)
- [← Back to Taxonomy](00-TAXONOMY.md)

**Version:** 1.0
**Last Updated:** 2025-10-03
**Maintainer:** ProduktRAG Project
