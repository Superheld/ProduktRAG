# Model Taxonomy - Der komplette ML Model Zoo

**Zweck:** Vollständiger Überblick über die ML-Model-Landschaft
**Zielgruppe:** Entwickler die verstehen wollen, welche Models es gibt und wofür
**Scope:** Von Embeddings bis Multimodal, von Basics bis Production

---

## 📋 Table of Contents

1. [Text Models](#1-text-models)
2. [Vision Models](#2-vision-models)
3. [Audio Models](#3-audio-models)
4. [Video Models](#4-video-models)
5. [Multimodal Models](#5-multimodal-models)
6. [Model Infrastructure](#6-model-infrastructure)
7. [Quick Reference](#quick-reference)

---

## 1. Text Models

### 1.1 Embedding Models (Text → Vector)

**Purpose:** Text in numerische Vektoren umwandeln für semantische Suche

**Architekturen:**
- **BERT-based:** Bidirektional, gut für Similarity
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
  - `intfloat/multilingual-e5-large` (1024 dim)
  - `deepset/gbert-large` (768 dim, Deutsch)

- **Contrastive Learning:**
  - `OpenAI/text-embedding-3-large` (3072 dim, API)
  - `Cohere/embed-multilingual-v3.0` (1024 dim, API)

**Deep-Dive:** Siehe [01-EMBEDDING-MODELS.md](01-EMBEDDING-MODELS.md)

---

### 1.2 Large Language Models (Text → Text)

**Purpose:** Text generieren, Chat, Completion, Reasoning

**Model Families:**

#### **GPT-Familie (Decoder-Only)**
- **GPT-4** (OpenAI, API, proprietär)
  - 1.7T Parameter (gemunkelt)
  - 128k Context Window
  - Multimodal (Vision)

- **GPT-3.5-Turbo** (OpenAI, API)
  - 175B Parameter
  - 16k Context Window
  - Schneller, günstiger

#### **Llama-Familie (Open Source, Meta)**
- **Llama 3.1** (8B, 70B, 405B)
  - 128k Context Window
  - Sehr gut für Reasoning
  - Basis für viele Forks

- **Llama 2** (7B, 13B, 70B)
  - 4k Context Window
  - Vorgänger, aber immer noch gut

#### **Mistral-Familie (Open Source, Mistral AI)**
- **Mistral 7B v0.3**
  - 7B Parameter
  - 32k Context Window
  - Sehr effizient (besser als Llama 2 13B)

- **Mixtral 8x7B** (Mixture of Experts)
  - 47B Parameter (8 Experten à 7B, nur 2 aktiv pro Token)
  - 32k Context Window
  - Schneller als 47B Dense Model

- **Mistral Large** (API)
  - Proprietär
  - Konkurrenz zu GPT-4

#### **Gemma-Familie (Google)**
- **Gemma 2** (9B, 27B)
  - Sehr schnell
  - Gute Instruktionsbefolgung
  - Lizenz: Kommerziell nutzbar

#### **Phi-Familie (Microsoft)**
- **Phi-3** (3.8B, 7B, 14B)
  - Sehr klein, aber stark
  - Gut für Edge/Mobile
  - 128k Context Window

#### **Claude (Anthropic, API)**
- **Claude 3.5 Sonnet**
  - Multimodal
  - 200k Context Window
  - Sehr gut für Code

**Deep-Dive:** Siehe [02-LLM-ARCHITECTURES.md](02-LLM-ARCHITECTURES.md)

---

### 1.3 Specialized Text Models

#### **Summarization**
- **BART** (Facebook)
- **T5** (Google)
- **Pegasus** (Google, News-spezifisch)

#### **Translation**
- **MarianMT** (Helsinki-NLP, viele Sprachpaare)
- **NLLB** (Meta, 200 Sprachen)
- **Opus-MT** (Community)

#### **Named Entity Recognition (NER)**
- **spaCy Models** (de_core_news_lg, en_core_web_trf)
- **Flair** (Character-level)
- **BERT-based NER** (Fine-tuned)

#### **Classification**
- **DistilBERT** (Schnell)
- **RoBERTa** (Robust)
- **DeBERTa** (State-of-the-art)

**Deep-Dive:** Siehe [03-SPECIALIZED-TEXT-MODELS.md](03-SPECIALIZED-TEXT-MODELS.md)

---

## 2. Vision Models

### 2.1 Image Classification

**Purpose:** Bild → Klasse/Label

**Klassische CNNs:**
- **ResNet** (50, 101, 152 Layer)
  - Standard für lange Zeit
  - Residual Connections

- **EfficientNet** (B0-B7)
  - Optimal skaliert (Depth, Width, Resolution)
  - Schneller als ResNet bei gleicher Accuracy

**Vision Transformers:**
- **ViT** (Vision Transformer)
  - Patcht Bild in 16x16 Quadrate
  - Transformer statt CNN
  - Braucht viele Daten

- **DeiT** (Data-efficient ViT)
  - Weniger Daten nötig
  - Knowledge Distillation

**Use Cases:** Produkt-Erkennung, Qualitätskontrolle, Medizinische Bildgebung

---

### 2.2 Object Detection

**Purpose:** Bild → Bounding Boxes + Labels

**YOLO-Familie (You Only Look Once)**
- **YOLOv8** (Ultralytics, aktuell)
  - Sehr schnell (Real-time)
  - Gut für Edge
  - Nano, Small, Medium, Large, XLarge Varianten

- **YOLOv10** (neueste, 2024)
  - Noch schneller
  - NMS-free (kein Non-Maximum Suppression nötig)

**Andere:**
- **Faster R-CNN** (Langsamer, aber akkurat)
- **DETR** (Detection Transformer, Facebook)
- **SAM** (Segment Anything Model, Meta)
  - Zero-shot Segmentation
  - "Clicke auf Objekt → wird segmentiert"

**Use Cases:** Autonomes Fahren, Überwachung, Retail

---

### 2.3 Image Generation

**Diffusion Models:**
- **Stable Diffusion** (Stability AI)
  - Open Source
  - Text → Image
  - Lokal ausführbar (VRAM: 6-12GB)
  - Versionen: 1.5, 2.1, XL, 3

- **DALL-E 3** (OpenAI, API)
  - Proprietär
  - Sehr gut für Text-Rendering

**GANs (älter, weniger relevant):**
- **StyleGAN** (NVIDIA)
- **BigGAN** (Google)

**Use Cases:** Marketing, Prototyping, Kunst

---

### 2.4 OCR & Document Understanding

**OCR (Optical Character Recognition):**
- **Tesseract** (Open Source, klassisch)
- **EasyOCR** (Deep Learning-based)
- **PaddleOCR** (Baidu, sehr gut)
- **Textract** (AWS, API)

**Document Understanding:**
- **LayoutLM** (Microsoft)
  - Versteht Layout + Text
  - Für Formulare, Invoices

- **Donut** (Document Understanding Transformer)
  - End-to-End (Bild → strukturierte Daten)
  - Kein OCR-Zwischenschritt nötig

**Use Cases:** Invoice Processing, Form Extraction, Archivierung

**Deep-Dive:** Siehe [04-VISION-MODELS.md](04-VISION-MODELS.md)

---

## 3. Audio Models

### 3.1 Speech-to-Text (ASR)

**Whisper (OpenAI)**
- 5 Größen: Tiny, Base, Small, Medium, Large
- 99 Sprachen
- Sehr robust gegen Hintergrund-Noise
- Open Source
- `whisper-large-v3` (aktuell)

**Andere:**
- **Wav2Vec 2.0** (Meta)
- **Conformer** (Google)
- **AssemblyAI** (API)

**Use Cases:** Transkription, Untertitel, Voice Assistants

---

### 3.2 Text-to-Speech (TTS)

**Moderne Models:**
- **Bark** (Suno AI)
  - Multilingual
  - Emotional Speech
  - Lachen, Seufzen möglich

- **XTTS** (Coqui)
  - Voice Cloning (13 Sekunden Audio)
  - Multilingual

- **VITS** (Conditional Variational Autoencoder)
  - Schnell
  - Gute Qualität

**Commercial:**
- **ElevenLabs** (API, beste Qualität)
- **Google Cloud TTS**
- **Azure Speech**

**Use Cases:** Hörbücher, Accessibility, Voice Bots

---

### 3.3 Audio Classification & Separation

**Classification:**
- **YAMNet** (Google, Audio Event Detection)
- **Wav2Vec 2.0** (Fine-tuned für Klassifikation)

**Separation:**
- **Spleeter** (Deezer)
  - Vocals vs Instruments
- **Demucs** (Meta)
  - State-of-the-art

**Deep-Dive:** Siehe [05-AUDIO-MODELS.md](05-AUDIO-MODELS.md)

---

## 4. Video Models

### 4.1 Action Recognition

**Purpose:** Video → Aktion/Ereignis

**Models:**
- **I3D** (Inflated 3D ConvNet)
- **SlowFast** (Meta)
  - Zwei Paths: Slow (Spatial), Fast (Temporal)
- **VideoMAE** (Masked Autoencoders)

**Use Cases:** Überwachung, Sport-Analyse, Gesten-Erkennung

---

### 4.2 Video Generation

**Text-to-Video:**
- **Runway Gen-2** (API)
- **Pika Labs** (Web)
- **Stable Video Diffusion** (Stability AI)
  - Open Source
  - Bild → kurzes Video

**Use Cases:** Marketing, Prototyping, Content Creation

---

### 4.3 Video Understanding

**Models:**
- **VideoLLaMA** (Multimodal LLM + Video)
- **Video-ChatGPT** (Video-Fragen beantworten)

**Deep-Dive:** Siehe [06-VIDEO-MODELS.md](06-VIDEO-MODELS.md)

---

## 5. Multimodal Models

### 5.1 Vision-Language Models

**CLIP (Contrastive Language-Image Pre-training, OpenAI)**
- Text + Bild in gemeinsamen Embedding-Space
- Zero-shot Image Classification
- Basis für viele Anwendungen (Stable Diffusion, Image Search)
- Varianten: `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`

**BLIP (Bootstrapped Language-Image Pre-training, Salesforce)**
- Image Captioning
- Visual Question Answering (VQA)
- Image-Text Retrieval

**LLaVA (Large Language and Vision Assistant)**
- Llama + Vision Encoder
- "GPT-4V für Arme" (Open Source)
- Kann Bilder beschreiben, analysieren
- `llava-v1.6-vicuna-13b`

**GPT-4 Vision (OpenAI, API)**
- Proprietär
- State-of-the-art
- PDF, Screenshots, Diagramme verstehen

**Gemini (Google, API)**
- Multimodal (Text, Image, Video, Audio)
- Native Multimodal (nicht nachträglich kombiniert)
- 1M+ Token Context Window (Gemini 1.5 Pro)

**Use Cases:**
- Image Search (CLIP)
- Visual Question Answering (LLaVA, GPT-4V)
- Document Understanding (GPT-4V + OCR)
- Accessibility (Bild-Beschreibungen)

---

### 5.2 Audio-Language Models

**Whisper + LLM Pipeline:**
- Audio → Text (Whisper) → LLM (Reasoning)

**Native Multimodal:**
- **Gemini** (kann direkt Audio verarbeiten)
- **GPT-4o** (Omni, Audio-nativ)

---

### 5.3 Any-to-Any Models

**ImageBind (Meta)**
- 6 Modalitäten: Image, Text, Audio, Depth, Thermal, IMU
- Gemeinsamer Embedding-Space
- Noch Research, nicht Production-ready

**NExT-GPT**
- Text, Image, Audio, Video
- Inputs + Outputs

**Deep-Dive:** Siehe [07-MULTIMODAL-MODELS.md](07-MULTIMODAL-MODELS.md)

---

## 6. Model Infrastructure

### 6.1 Model Serving & Inference

**Ollama**
- Lokal, einfach
- LLMs (Llama, Mistral, Gemma, etc.)
- Embeddings (nomic-embed-text, mxbai-embed-large)
- GGUF-Format (quantisiert)
- CLI + REST API

**vLLM**
- Production-Grade LLM Serving
- PagedAttention (effizientes KV-Cache Management)
- Höherer Throughput als Ollama
- OpenAI-kompatible API

**Text Generation Inference (TGI, Hugging Face)**
- Production Serving für Hugging Face Models
- Tensor Parallelism (Multi-GPU)
- Flash Attention, Paged Attention

**llama.cpp**
- Pure C++ Implementierung
- Sehr schnell
- Basis für Ollama
- GGUF-Format

**TensorRT-LLM (NVIDIA)**
- Optimiert für NVIDIA GPUs
- Extrem schnell
- Komplex zu setup

**Deep-Dive:** Siehe [08-MODEL-SERVING.md](08-MODEL-SERVING.md)

---

### 6.2 Quantization (Model komprimieren)

**Purpose:** Model kleiner machen bei minimalem Qualitätsverlust

**Formate:**

**GGUF (llama.cpp, Ollama)**
- CPU-optimiert
- 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, 8-bit
- `Q4_K_M` = 4-bit, K-Quantization, Medium
- Ollama nutzt das

**GPTQ (GPU-optimiert)**
- 2-bit, 3-bit, 4-bit, 8-bit
- Schneller auf GPU als GGUF
- `TheBloke/Llama-2-7B-GPTQ`

**AWQ (Activation-aware Weight Quantization)**
- Noch besser als GPTQ (weniger Qualitätsverlust)
- 4-bit
- vLLM, TGI Support

**bitsandbytes (Hugging Face)**
- 8-bit, 4-bit
- Für Training + Inference
- Basis für QLoRA

**Beispiel:**
- **Llama 2 7B:** 14GB (FP16) → 3.5GB (4-bit) = 75% kleiner
- **Qualitätsverlust:** ~1-2% auf Benchmarks

**Deep-Dive:** Siehe [09-QUANTIZATION.md](09-QUANTIZATION.md)

---

### 6.3 Model Formats

**SafeTensors**
- Sicherer als Pickle (kein Code-Execution)
- Schneller zu laden
- Hugging Face Standard

**ONNX (Open Neural Network Exchange)**
- Framework-agnostisch (PyTorch, TensorFlow, etc.)
- Optimiert für Inference
- ONNX Runtime sehr schnell

**TensorRT**
- NVIDIA-spezifisch
- Extrem optimiert
- FP16, INT8 Support

**CoreML (Apple)**
- iOS/macOS
- Neural Engine Support (M1/M2/M3 Chips)

**TensorFlow Lite**
- Mobile (Android/iOS)
- Edge Devices

**Deep-Dive:** Siehe [10-MODEL-FORMATS.md](10-MODEL-FORMATS.md)

---

### 6.4 Fine-Tuning Methods

**Full Fine-Tuning**
- Alle Parameter updaten
- Braucht viel VRAM
- Beste Qualität

**LoRA (Low-Rank Adaptation)**
- Nur kleine Matrizen trainieren
- ~0.1% der Parameter
- 1-2GB VRAM für 7B Model
- 95% der Full Fine-Tuning Qualität

**QLoRA (Quantized LoRA)**
- LoRA + 4-bit Base Model
- 7B Model trainieren mit 6GB VRAM
- State-of-the-art für Consumer Hardware

**Prefix Tuning / Prompt Tuning**
- Nur Prompt-Embeddings trainieren
- Noch speichereffizienter
- Etwas schlechter als LoRA

**Adapters**
- Kleine Layer zwischen Transformer-Blocks
- Mehrere Tasks mit einem Base-Model

**RLHF (Reinforcement Learning from Human Feedback)**
- Wie ChatGPT trainiert wurde
- Model lernt Präferenzen
- Sehr aufwendig

**DPO (Direct Preference Optimization)**
- Einfacher als RLHF
- Gleiche Ziele
- Weniger Resourcen

**Deep-Dive:** Siehe [11-FINE-TUNING.md](11-FINE-TUNING.md)

---

## Navigation

- **[01-EMBEDDING-MODELS.md](01-EMBEDDING-MODELS.md)** - Deep-Dive in Embedding-Architekturen
- **[02-LLM-ARCHITECTURES.md](02-LLM-ARCHITECTURES.md)** - LLM-Familien, Architekturen, Context Window
- **[03-SPECIALIZED-TEXT-MODELS.md](03-SPECIALIZED-TEXT-MODELS.md)** - NER, Translation, Summarization
- **[04-VISION-MODELS.md](04-VISION-MODELS.md)** - CNNs, ViT, YOLO, SAM, Stable Diffusion
- **[05-AUDIO-MODELS.md](05-AUDIO-MODELS.md)** - Whisper, TTS, Audio Classification
- **[06-VIDEO-MODELS.md](06-VIDEO-MODELS.md)** - Action Recognition, Video Generation
- **[07-MULTIMODAL-MODELS.md](07-MULTIMODAL-MODELS.md)** - CLIP, LLaVA, GPT-4V, Gemini
- **[08-MODEL-SERVING.md](08-MODEL-SERVING.md)** - Ollama, vLLM, TGI, llama.cpp
- **[09-QUANTIZATION.md](09-QUANTIZATION.md)** - GGUF, GPTQ, AWQ, bitsandbytes
- **[10-MODEL-FORMATS.md](10-MODEL-FORMATS.md)** - ONNX, TensorRT, CoreML, SafeTensors
- **[11-FINE-TUNING.md](11-FINE-TUNING.md)** - LoRA, QLoRA, RLHF, DPO

---

**Version:** 1.0
**Last Updated:** 2025-10-03
**Maintainer:** ProduktRAG Project
