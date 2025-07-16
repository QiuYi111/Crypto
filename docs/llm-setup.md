# LLM Setup and Configuration Guide

This guide covers setting up and configuring the Large Language Model integration for CryptoRL sentiment analysis.

## Overview

The LLM integration provides:
- **Sentiment Analysis**: 7-dimensional confidence vectors
- **News Processing**: Real-time and historical news analysis
- **Market Insights**: Contextual understanding of market events
- **Risk Assessment**: Dynamic risk scoring

## Supported Models

### Local Models (Recommended)

#### Llama 2/3 Family
```bash
# Install Llama 2 7B Chat
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# Or Llama 3 8B
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

#### Alternative Models
- **Mistral 7B**: Fast and efficient
- **Gemma 7B**: Google's open model
- **Mixtral 8x7B**: Mixture of experts
- **Quantized versions**: 4-bit/8-bit for lower memory usage

### Cloud Models

#### OpenAI
```bash
# Set in .env
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
```

#### Anthropic Claude
```bash
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku-20240307
```

## Installation

### 1. Local LLM Setup

#### Using Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2:7b-chat
ollama pull mistral:7b
ollama pull codellama:7b-code

# Test installation
ollama run llama2:7b-chat "What is the sentiment of Bitcoin today?"
```

#### Using Hugging Face Transformers
```bash
# Install with GPU support
pip install transformers torch accelerate

# For 4-bit quantization
pip install bitsandbytes

# Install sentence transformers for embeddings
pip install sentence-transformers
```

#### Using Llama.cpp
```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Download GGUF model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Set model path in .env
LLM_MODEL_PATH=/path/to/llama-2-7b-chat.Q4_K_M.gguf
```

### 2. Configuration

#### Environment Variables

```bash
# Local model configuration
LLM_TYPE=local
LLM_MODEL_PATH=/models/llama-2-7b-chat
LLM_DEVICE=cuda  # or cpu
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7
LLM_BATCH_SIZE=4

# For Ollama
LLM_TYPE=ollama
LLM_MODEL_NAME=llama2:7b-chat
OLLAMA_URL=http://localhost:11434

# For OpenAI
LLM_TYPE=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-3.5-turbo
```

#### Model Configuration File

Create `config/llm_config.yaml`:

```yaml
models:
  llama2_7b:
    type: huggingface
    model_name: meta-llama/Llama-2-7b-chat-hf
    device: cuda
    max_tokens: 512
    temperature: 0.7
    top_p: 0.9
    
  mistral_7b:
    type: ollama
    model_name: mistral:7b
    base_url: http://localhost:11434
    
  gpt35:
    type: openai
    model_name: gpt-3.5-turbo
    api_key: ${OPENAI_API_KEY}

confidence_vectors:
  dimensions:
    - fundamentals
    - industry_condition
    - geopolitics
    - macroeconomics
    - technical_analysis
    - market_sentiment
    - risk_assessment
  
  prompts:
    system: |
      You are a cryptocurrency market analyst. Analyze the provided information and generate a confidence score for each dimension.
      
    user_template: |
      Analyze {symbol} for {date}:
      {context}
      
      Provide confidence scores (0-1) for:
      1. Fundamentals
      2. Industry condition  
      3. Geopolitics
      4. Macroeconomics
      5. Technical analysis
      6. Market sentiment
      7. Risk assessment
```

## Usage Examples

### Basic Sentiment Analysis

```python
from cryptorl.llm import LLMClient, ConfidenceGenerator

# Initialize client
llm = LLMClient(
    model_type="ollama",
    model_name="llama2:7b-chat"
)

# Generate confidence vector
generator = ConfidenceGenerator(llm)
confidence = generator.analyze_symbol(
    symbol="BTCUSDT",
    date="2024-01-15",
    news_context=True
)

print(confidence)
# Output: [0.7, 0.6, 0.4, 0.8, 0.5, 0.6, 0.3]
```

### Batch Processing

```python
from cryptorl.llm import BatchProcessor

processor = BatchProcessor(
    model_type="huggingface",
    model_name="meta-llama/Llama-2-7b-chat-hf"
)

# Process multiple symbols
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
dates = ["2024-01-14", "2024-01-15", "2024-01-16"]

results = processor.batch_analyze(
    symbols=symbols,
    dates=dates,
    max_workers=4
)
```

### Custom Prompts

```python
from cryptorl.llm import PromptBuilder

builder = PromptBuilder()
prompt = builder.create_prompt(
    symbol="BTCUSDT",
    date="2024-01-15",
    context={
        "price_change": "+5.2%",
        "volume": "High",
        "news": ["ETF approval rumors", "Regulatory clarity"],
        "technical": "Breaking resistance at $45k"
    }
)

response = llm.generate(prompt)
```

## Performance Optimization

### 1. Model Quantization

```python
# 4-bit quantization for lower memory usage
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 2. Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedLLM:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    @lru_cache(maxsize=1000)
    def generate_confidence(self, symbol, date, context_hash):
        return self.llm.analyze(symbol, date, context_hash)
```

### 3. Batch Processing

```python
# Efficient batch processing
class BatchLLM:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
    
    def process_batch(self, prompts):
        results = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            batch_results = self.model.generate_batch(batch)
            results.extend(batch_results)
        return results
```

## Testing and Validation

### 1. Model Performance Testing

```python
import time
from cryptorl.llm import LLMTester

tester = LLMTester()

# Test inference speed
start_time = time.time()
result = llm.generate("Analyze BTC sentiment")
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f}s")

# Test accuracy
accuracy = tester.validate_confidence_accuracy(
    test_data="data/test_confidence.json",
    model=llm
)
print(f"Accuracy: {accuracy:.2%}")
```

### 2. A/B Testing Models

```python
from cryptorl.llm import ModelComparator

comparator = ModelComparator()

results = comparator.compare_models(
    models=["llama2:7b", "mistral:7b", "gpt-3.5-turbo"],
    test_cases="data/sentiment_test.json",
    metrics=["latency", "accuracy", "cost"]
)

comparator.plot_results(results)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
export LLM_BATCH_SIZE=1

# Use CPU instead
export LLM_DEVICE=cpu

# Try quantized model
export LLM_MODEL_PATH=/models/llama-2-7b-chat-q4.gguf
```

#### 2. Model Loading Issues
```bash
# Check model path
ls -la $LLM_MODEL_PATH

# Verify Hugging Face token
huggingface-cli login

# Check disk space
df -h
```

#### 3. Performance Issues
```bash
# Monitor GPU usage
nvidia-smi

# Check CPU usage
htop

# Profile memory usage
python -m memory_profiler llm_client.py
```

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
from cryptorl.llm import set_log_level
set_log_level("DEBUG")

# Test model directly
llm = LLMClient()
response = llm.generate("Test prompt", max_tokens=10)
print(response)
```

## Advanced Features

### 1. Fine-tuning

```python
from cryptorl.llm import FinetuneTrainer

trainer = FinetuneTrainer(
    base_model="meta-llama/Llama-2-7b-chat-hf",
    dataset_path="data/crypto_sentiment.jsonl"
)

trainer.train(
    output_dir="models/finetuned-crypto-llama",
    epochs=3,
    learning_rate=2e-4
)
```

### 2. Custom Embeddings

```python
from sentence_transformers import SentenceTransformer

class CryptoEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_news(self, news_text):
        return self.model.encode(news_text)
    
    def similarity_search(self, query_embedding, news_embeddings):
        return cosine_similarity([query_embedding], news_embeddings)
```

### 3. Multi-modal Analysis

```python
from cryptorl.llm import MultiModalAnalyzer

analyzer = MultiModalAnalyzer()

# Analyze news with charts
confidence = analyzer.analyze_with_chart(
    symbol="BTCUSDT",
    news_text="Bitcoin breaks $50k resistance",
    chart_path="charts/btc_1h.png"
)
```

## Monitoring and Metrics

### 1. Model Performance

```python
from cryptorl.llm import PerformanceMonitor

monitor = PerformanceMonitor()

# Track metrics
metrics = monitor.track_metrics(
    model_name="llama2:7b",
    requests_per_minute=60,
    avg_latency=0.5,
    accuracy=0.85
)

# Generate report
report = monitor.generate_report()
monitor.save_report("reports/llm_performance.json")
```

### 2. Cost Tracking

```python
from cryptorl.llm import CostTracker

tracker = CostTracker()

# Track API costs
tracker.track_request(
    model="gpt-3.5-turbo",
    tokens=150,
    cost=0.002
)

monthly_cost = tracker.get_monthly_cost()
print(f"Monthly LLM cost: ${monthly_cost:.2f}")
```