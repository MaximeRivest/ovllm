# OvLLM 🚀

> This is the very first 'working' version. Expect things to change and improve a lot. Still, what is documented here works on my machine.

**One-line vLLM for everyone**

OvLLM is a Python library that makes running local LLMs as easy as a function call, while leveraging the incredible performance of [vLLM](https://github.com/vllm-project/vllm). It's designed for simplicity without sacrificing power, featuring native [DSPy](https://github.com/stanfordnlp/dspy) integration with proper output formatting and automatic request batching for maximum GPU efficiency.

At its core, `ovllm` ensures you can get started instantly and build complex pipelines without worrying about the underlying engine's state management, memory, or batching logic.

-----

## ✨ Features

  - **Zero-Config Startup**: Works out of the box with a sensible default model that runs on most systems.
  - **One-Line Model Swapping**: Hot-swap models on the GPU with a single command: `llmtogpu("any-huggingface-model")`.
  - **Automatic Request Batching**: Transparently groups concurrent requests for optimal GPU throughput.
  - **Native DSPy Compatibility**: The `llm` object is a first-class `dspy.LM`, ready for any DSPy module or optimizer.
  - **Smart Memory Management**: Automatically unloads old models and clears GPU memory when you switch.
  - **Helpful Errors & Helpers**: Get clear, actionable error messages and use helpers like `suggest_models()` to find the right model for your hardware.
  - **Rich Documentation**: Comprehensive help is built-in via Python's `help()` function.

-----

## 📦 Installation

**Prerequisite**: OVLLM uses vLLM, which requires an NVIDIA GPU with CUDA 12.1 or newer. Please ensure your environment is set up correctly.

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

-----

## 🎯 Quick Start

### Basic Usage

Just import the `llm` object and call it. The first time you do, a small, capable default model will be downloaded and loaded into your GPU.

```python
import ovllm
# The first call loads the default model (Qwen/Qwen3-0.6B). Please wait a moment.
response = ovllm.llm("What is the capital of Canada?")

```

> **Note:** For full DSPy compatibility, `llm()` returns a list containing a completion object. That's why we access `response[0]` to get the first result.

### Switching Models

Easily switch to any model on the Hugging Face Hub. `ovllm` handles the cleanup automatically.

```python
from ovllm import llmtogpu, suggest_models, llm

# See what models your GPU can handle
suggest_models() # good suggestions to come! wip :)

# Load a different model
llmtogpu("google/gemma-3n-E4B-it", vllm_args={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.80}) 

# Now all calls use the new model
response = llm("Explain quantum computing in simple terms")
print(response[0])
```

-----

## 🤖 DSPy Integration

OVLLM is designed to be a perfect companion for DSPy. Just configure it once.

### Simple Prediction

```python
import dspy
import ovllm

# Configure DSPy to use your local OVLLM instance
dspy.configure(lm=ovllm.llm)

# Create a simple predictor
predict = dspy.Predict("question -> answer")

# Run the predictor
result = predict(question="What is the powerhouse of the cell?")
print(result.answer)
```

### Chain of Thought (CoT) Reasoning

```python
import dspy
import ovllm

dspy.configure(lm=ovllm.llm)

# Use ChainOfThought to encourage step-by-step reasoning
cot_predictor = dspy.ChainOfThought("question -> answer")

result = cot_predictor(question="If I have 5 apples and I eat 2, then buy 3 more, how many apples do I have left?")
print(f"Answer: {result.answer}")
# The model's reasoning is also available!
print(f"\nReasoning:\n{result.reasoning=}")
```

### Automatic Batching

When you use DSPy features that make multiple calls to the LM (like `predict.batch` or optimizers), OVLLM's `AutoBatchLM` layer automatically catches these concurrent requests and sends them to the GPU in a single, efficient batch. You don't have to do anything extra to get this performance boost.

```python
import dspy
import ovllm

dspy.configure(lm=ovllm.llm)

questions = [
    "What color is the sky on a clear day?",
    "What is 2+2?",
    "What is the main component of air?",
]

examples = [dspy.Example(question=q).with_inputs("question") for q in questions]

predict = dspy.Predict("question -> answer")

# This automatically runs as a single efficient batch on the GPU!
results = predict.batch(examples)

for ex, res in zip(examples, results):
    print(f"Q: {ex.question}")
    print(f"A: {res.answer}\n")
```

-----

## 🛠️ Advanced Usage

### Custom Parameters

Pass any vLLM-supported parameters directly to `llmtogpu` to customize model loading and generation.

```python
from ovllm import llmtogpu

llmtogpu(
    "microsoft/phi-2",
    temperature=0.0,                  # For deterministic outputs
    max_tokens=2048,                  # Allow longer responses
    gpu_memory_utilization=0.9,       # Use 90% of GPU VRAM
    dtype="float16"                   # Use specific precision
)
```

### Error Handling

OVLLM provides clear, actionable error messages if something goes wrong.

**Model too large for VRAM:**

```
❌ ERROR: Not enough GPU memory to load 'meta-llama/Llama-2-70b-hf'.
   Try lowering the `gpu_memory_utilization` (e.g., `llmtogpu(..., gpu_memory_utilization=0.8)`) or use a smaller model.
```

**Gated Hugging Face Model:**

```
❌ ERROR: The HuggingFace repository for 'meta-llama/Llama-3-8B-Instruct' is gated.
   1. Visit https://huggingface.co/settings/tokens to create a token.
   2. Run `huggingface-cli login` in your terminal and paste the token.
```

-----

## ⚙️ How It Works

OVLLM's simplicity is enabled by a robust underlying architecture designed to solve common challenges with local LLMs.

1.  **The Proxy Object (`llm`)**: When you use `ovllm.llm`, you're interacting with a lightweight proxy object. This object doesn't contain the massive model itself, so it can be safely copied by DSPy without duplicating the engine.
2.  **The Singleton Manager**: The proxy communicates with a single, global instance manager. On the first call, this manager loads the vLLM engine into the GPU.
3.  **The Auto-Batching Queue (`AutoBatchLM`)**: All requests from the proxy are sent to an intelligent queue. This queue collects concurrent requests and groups them into an optimal batch before sending them to the vLLM engine, maximizing GPU throughput.
4.  **Automatic Cleanup**: When you call `llmtogpu()`, the manager gracefully shuts down the old engine and its batching queue, clears the GPU memory, and then loads the new model.

This architecture gives you the best of both worlds: a dead-simple, stateless-feeling API and a high-performance, statefully-managed backend.

-----

## 🤝 Contributing

Contributions are welcome\! If you find a bug or have a feature request, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.