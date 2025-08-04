# OVLLM 🚀

**One-line vLLM for everyone**

OVLLM is a Python library that makes running local LLMs as easy as `llm("hello")` while leveraging the incredible performance of [vLLM](https://github.com/vllm-project/vllm). It's designed for simplicity without sacrificing power, featuring seamless [DSPy](https://github.com/stanfordnlp/dspy) integration and automatic request batching for maximum GPU efficiency.

## ✨ Features

- **Zero-config startup**: Works out of the box with a sensible default model
- **One-line model switching**: `llmtogpu("any-huggingface-model")` 
- **Automatic batching**: Transparently groups requests for optimal GPU utilization
- **DSPy native**: First-class support for DSPy pipelines
- **Smart memory management**: Automatically cleans up when switching models
- **Helpful errors**: Clear messages when models are too large for your GPU
- **Rich documentation**: Comprehensive help via `help(ovllm)`


## 📦 Installation

### From PyPI (Recommended)

```bash
pip install ovllm
```

### From Source

```bash
git clone https://github.com/maximerivest/ovllm
cd ovllm
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/maximerivest/ovllm
cd ovllm
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage

```python
from ovllm import llm

# Just works - uses default model (Qwen/Qwen2.5-0.5B-Instruct)
response = llm("What is the capital of France?")
print(response)  # "The capital of France is Paris."
```

### Switching Models

```python
from ovllm import llmtogpu, suggest_models

# See what models your GPU can handle
suggest_models()

# Load a different model (automatic cleanup of previous model)
llmtogpu("google/gemma-2b-it")

# Now all calls use the new model
response = llm("Explain quantum computing in simple terms")
print(response)
```

## 🤖 DSPy Integration

OVLLM is designed from the ground up to work seamlessly with DSPy:

### Simple DSPy Usage

```python
import dspy
import ovllm

# Configure DSPy to use OVLLM
dspy.configure(lm=ovllm.llm)

# Create a simple prediction
predict = dspy.Predict("question -> answer")
result = predict(question="What is the meaning of life?")
print(result.answer)
```

### Advanced DSPy with Context

```python
import dspy
import ovllm

# Load a larger model for more complex tasks
ovllm.llmtogpu("Qwen/Qwen2.5-3B-Instruct")
dspy.configure(lm=ovllm.llm)

# Create a RAG-style predictor
rag = dspy.Predict("question, context -> answer")
result = rag(
    question="How old is the king of England?",
    context="King Charles III was born on November 14, 1948."
)
print(result.answer)
```

### Batch Processing (Automatic Batching!)

OVLLM automatically batches requests for maximum GPU efficiency:

```python
import dspy, ovllm

ovllm.llmtogpu("Qwen/Qwen2.5-1.5B-Instruct")
dspy.configure(lm=ovllm.llm)

# Create examples
examples = [
    dspy.Example(question="What is AI?", context="AI is artificial intelligence."),
    dspy.Example(question="Capital of Japan?", context="The capital is Tokyo."),
    dspy.Example(question="Who wrote Python?", context="Python was created by Guido van Rossum."),
]

examples = [ex.with_inputs("question", "context") for ex in examples]

# This automatically batches all requests together!
predict = dspy.Predict("question, context -> answer")
results = predict.batch(examples)

for result in results:
    print(result.answer)
```

## 🛠️ Advanced Features

### GPU Memory Management

```python
from ovllm import get_gpu_memory, suggest_models

# Check available GPU memory
print(f"GPU Memory: {get_gpu_memory():.1f} GB")

# Get model recommendations
suggest_models()
# Output:
# Available GPU memory: 16.0 GB
# Suggested models for your system:
#   - Qwen/Qwen2.5-3B-Instruct (3B params, ~6GB VRAM)
#   - Qwen/Qwen2.5-7B-Instruct (7B params, ~14GB VRAM)
#   - meta-llama/Llama-3.2-3B-Instruct (3B params, ~6GB VRAM)
```

### Custom Parameters

```python
# Load model with custom parameters
llmtogpu(
    "microsoft/phi-2",
    temperature=0.0,      # Deterministic outputs
    max_tokens=1024,      # Longer responses
    dtype="float16"       # Specific precision
)
```

### Error Handling

OVLLM provides clear, actionable error messages:

```python
# If you try to load a model that's too large
llmtogpu("meta-llama/Llama-2-70b-hf")
# Error: Not enough GPU memory to load meta-llama/Llama-2-70b-hf.
# Try a smaller model like:
#   - Qwen/Qwen2.5-0.5B-Instruct (0.5B parameters)
#   - Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters)
#   - google/gemma-2b-it (2B parameters)
```

## 📚 Documentation

Get comprehensive help directly in Python:

```python
import ovllm

# Detailed help and examples
help(ovllm)

# Specific function help
help(ovllm.llmtogpu)
help(ovllm.llm)
```

## 🏗️ Architecture

OVLLM consists of three main components:

1. **VLLMChatLM**: A thin wrapper around vLLM that speaks DSPy's protocol
2. **AutoBatchLM**: An intelligent batching layer that accumulates requests
3. **GlobalLLM**: A singleton manager that handles model lifecycle

The library automatically:
- Loads a small default model on first use
- Batches concurrent requests for efficiency
- Cleans up GPU memory when switching models
- Provides helpful error messages and suggestions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the amazing [vLLM](https://github.com/vllm-project/vllm) project
- Designed for seamless integration with [DSPy](https://github.com/stanfordnlp/dspy)
- Inspired by the simplicity of [Ollama](https://ollama.ai/)