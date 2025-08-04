"""
OVLLM - One-line vLLM for local inference, with first-class DSPy support.

>>> import ovllm, dspy
>>> dspy.configure(lm = ovllm.llm)  # zero-boilerplate
>>> ovllm.llm("Hello!")             # or use inside DSPy programs
"""

from __future__ import annotations

import asyncio
import threading
import warnings
import torch
import gc
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from functools import wraps

import dspy
    
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "vLLM is required for OVLLM. Install it with:\n"
        "pip install vllm"
    )


# Default model that works on most machines
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512


def _wrap_request_output(o, model: str) -> SimpleNamespace:
    """Convert vLLM output to OpenAI-style format expected by DSPy."""
    comp = o.outputs[0]
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(
            index=0,
            finish_reason=getattr(comp, 'finish_reason', 'stop'),
            message=SimpleNamespace(content=comp.text),
        )],
        usage={
            "prompt_tokens": len(o.prompt_token_ids),
            "completion_tokens": len(comp.token_ids),
            "total_tokens": len(o.prompt_token_ids) + len(comp.token_ids),
        },
    )


class VLLMChatLM(dspy.BaseLM):
    """Offline vLLM engine that speaks DSPy's BaseLM protocol."""
    
    supports_batch = True
    
    def __init__(
        self,
        model: str,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        dtype: str = "auto",
        **sampler_overrides,
    ):
        """
        Initialize vLLM chat model.
        
        Args:
            model: HuggingFace model ID or local path
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            dtype: Model dtype ("auto", "float16", "bfloat16", etc.)
            **sampler_overrides: Additional vLLM sampling parameters
        """
        super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens)
        
        # Initialize vLLM engine
        try:
            self._engine = LLM(model=model, dtype=dtype, trust_remote_code=True)
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise MemoryError(
                    f"Not enough GPU memory to load {model}.\n"
                    f"Try a smaller model like:\n"
                    f"  - Qwen/Qwen2.5-0.5B-Instruct (0.5B parameters)\n"
                    f"  - Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters)\n"
                    f"  - google/gemma-2b-it (2B parameters)"
                ) from e
            elif "gated repo" in str(e).lower() or "401 client error" in str(e).lower():
                import os
                raise PermissionError(
                    f"\n{'='*60}\n"
                    f"⚠️  GATED MODEL ACCESS REQUIRED\n"
                    f"{'='*60}\n\n"
                    f"The model '{model}' requires authentication.\n\n"
                    f"To fix this:\n\n"
                    f"1. Get your HuggingFace token:\n"
                    f"   • Go to: https://huggingface.co/settings/tokens\n"
                    f"   • Create a new token with 'read' permissions\n"
                    f"   • Copy the token (starts with 'hf_...')\n\n"
                    f"2. Set your token (choose one method):\n\n"
                    f"   a) Set it permanently (recommended):\n"
                    f"      export HF_TOKEN='your_token_here'\n"
                    f"      # Add to ~/.bashrc or ~/.zshrc to persist\n\n"
                    f"   b) Use huggingface-cli:\n"
                    f"      huggingface-cli login\n"
                    f"      # Paste your token when prompted\n\n"
                    f"   c) Set it in Python:\n"
                    f"      import os\n"
                    f"      os.environ['HF_TOKEN'] = 'your_token_here'\n\n"
                    f"3. Request access to the model:\n"
                    f"   • Visit: https://huggingface.co/{model}\n"
                    f"   • Click 'Request access' if needed\n"
                    f"   • Wait for approval (usually instant)\n\n"
                    f"Current token status: "
                    f"{'✓ Token found' if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') else '✗ No token found'}\n"
                    f"{'='*60}"
                ) from e
            raise
            
        self._base_sampling = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            **sampler_overrides
        )
    
    def __call__(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kw):
        """Direct call interface for simple usage."""
        result = self.forward(prompt, messages, **kw)
        if hasattr(result, 'choices') and result.choices:
            return result.choices[0].message.content
        return str(result)
    
    def forward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kw):
        """Single request forward pass."""
        return self.forward_batch([prompt], [messages], **kw)[0]
    
    async def aforward(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kw):
        """Async single request forward pass."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.forward(prompt, messages, **kw)
        )
    
    def forward_batch(
        self,
        prompts: Sequence[str | None],
        messages_list: Sequence[List[Dict[str, str]] | None] | None = None,
        **kw,
    ):
        """Batch forward pass for multiple requests."""
        if messages_list is None:
            messages_list = [None] * len(prompts)
        
        norm_msgs: List[List[Dict[str, str]]] = []
        for p, m in zip(prompts, messages_list):
            norm_msgs.append(m if m is not None
                           else [{"role": "user", "content": p or ""}])
        
        sampling = SamplingParams(**{**self._base_sampling, **kw})
        raw = self._engine.chat(norm_msgs, sampling, use_tqdm=False)
        return [_wrap_request_output(o, self.model) for o in raw]
    
    def shutdown(self):
        """Clean shutdown of the vLLM engine."""
        if hasattr(self, '_engine') and self._engine is not None:
            try:
                # Try to shutdown gracefully
                if hasattr(self._engine, 'llm_engine'):
                    if hasattr(self._engine.llm_engine, 'engine_core'):
                        self._engine.llm_engine.engine_core.shutdown()
            except:
                pass
            del self._engine


class AutoBatchLM(dspy.BaseLM):
    """
    Intelligent batching wrapper for any LM backend.
    Accumulates requests and batches them for maximum GPU utilization.
    """
    
    supports_batch = True
    
    def __init__(
        self,
        backend: Any,
        *,
        max_batch: int = 128,
        flush_ms: int = 8,
    ):
        """
        Initialize auto-batching wrapper.
        
        Args:
            backend: The underlying LM to wrap
            max_batch: Maximum batch size before forcing flush
            flush_ms: Time in milliseconds to wait before flushing partial batch
        """
        # Get properties from backend before calling super().__init__
        model_type = getattr(backend, 'model_type', 'chat')
        temperature = getattr(backend, 'temperature', DEFAULT_TEMPERATURE)
        max_tokens = getattr(backend, 'max_tokens', DEFAULT_MAX_TOKENS)
        
        super().__init__(
            model=backend.model, 
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.backend = backend
        self.max_batch = max_batch
        self.flush_ms = flush_ms
        
        # Launch private event loop for batching
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._shutdown = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()
    
    def __call__(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kw):
        """Direct call interface."""
        return self.forward(prompt, messages, **kw)
    
    def forward(self, prompt=None, messages=None, **kw):
        """Enqueue request and wait for batched result."""
        if self._shutdown:
            raise RuntimeError("AutoBatchLM has been shut down")
        fut = asyncio.run_coroutine_threadsafe(
            self._enqueue(prompt, messages, kw),
            self._loop,
        )
        result = fut.result()
        
        # Return just the text for simple __call__ usage
        if hasattr(result, 'choices') and result.choices:
            return result.choices[0].message.content
        return result
    
    async def aforward(self, prompt=None, messages=None, **kw):
        """Async forward with batching."""
        if self._shutdown:
            raise RuntimeError("AutoBatchLM has been shut down")
        loop = asyncio.get_running_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._enqueue(prompt, messages, kw),
            self._loop,
        )
        return await asyncio.wrap_future(fut, loop=loop)
    
    def forward_batch(self, prompts, messages_list=None, **kw):
        """Direct batch forward (bypasses auto-batching)."""
        return self.backend.forward_batch(prompts, messages_list, **kw)
    
    async def _enqueue(self, p, m, kw):
        """Add request to queue and wait for result."""
        fut = self._loop.create_future()
        await self._q.put((p, m, kw, fut))
        return await fut
    
    def _run_loop(self):
        """Background thread running the batching event loop."""
        asyncio.set_event_loop(self._loop)
        self._q: asyncio.Queue = asyncio.Queue()
        self._ready.set()
        self._loop.create_task(self._collector())
        self._loop.run_forever()
    
    async def _collector(self):
        """Collect requests into batches and process them."""
        from asyncio import QueueEmpty
        while not self._shutdown:
            try:
                p, m, kw, fut = await asyncio.wait_for(self._q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
                
            bucket = [(p, m, kw, fut)]
            t0 = self._loop.time()
            
            # Collect more requests up to batch size or timeout
            while (len(bucket) < self.max_batch and
                   (self._loop.time() - t0) * 1_000 < self.flush_ms):
                try:
                    bucket.append(self._q.get_nowait())
                except QueueEmpty:
                    await asyncio.sleep(self.flush_ms / 4 / 1_000)
            
            # Group by kwargs for compatible batching
            by_kw: Dict[Tuple[Tuple[str, Any], ...], List[Tuple]] = defaultdict(list)
            for p, m, kw, fut in bucket:
                by_kw[tuple(sorted(kw.items()))].append((p, m, fut))
            
            # Process each compatible group
            for kw_key, grp in by_kw.items():
                p_list = [x[0] for x in grp]
                m_list = [x[1] for x in grp]
                kw_shared = dict(kw_key)
                try:
                    outs = self.backend.forward_batch(p_list, m_list, **kw_shared)
                    if len(outs) != len(grp):
                        raise RuntimeError("Backend returned mismatched #outputs")
                    for o, (_, _, f) in zip(outs, grp):
                        if not f.done():
                            f.set_result(o)
                except Exception as exc:
                    for _, _, f in grp:
                        if not f.done():
                            f.set_exception(exc)
    
    def shutdown(self):
        """Clean shutdown of batching system."""
        self._shutdown = True
        if hasattr(self.backend, 'shutdown'):
            self.backend.shutdown()


class GlobalLLM(dspy.BaseLM):
    """Global LLM singleton manager with DSPy compatibility."""
    
    _instance = None
    _backend = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            super().__init__(model=DEFAULT_MODEL, model_type="chat")
            self._initialized = True
            self._backend = None
    
    def _initialize_default(self):
        """Initialize with default small model."""
        self._load_model(DEFAULT_MODEL)
    
    def _save_token(self, token: str):
        """Save HuggingFace token to user's home directory."""
        import os
        from pathlib import Path
        
        # Save to HuggingFace's default location
        hf_dir = Path.home() / ".cache" / "huggingface"
        hf_dir.mkdir(parents=True, exist_ok=True)
        
        token_file = hf_dir / "token"
        try:
            token_file.write_text(token)
            token_file.chmod(0o600)  # Secure permissions
            print(f"✓ Token saved to {token_file}")
            print("  This will be used automatically by HuggingFace libraries.")
        except Exception as e:
            print(f"Warning: Could not save token to file: {e}")
            print("Token is still set for this session.")
    
    def _check_model_access(self, model_name: str):
        """Check if model requires authentication and help user set it up."""
        import os
        import getpass
        
        # Known gated models that require authentication
        gated_models = [
            "google/gemma", "meta-llama/Llama", "mistralai/Mistral", 
            "meta-llama/Meta-Llama", "microsoft/phi", "google/recurrentgemma"
        ]
        
        # Check if this might be a gated model
        is_likely_gated = any(model_name.startswith(prefix) for prefix in gated_models)
        
        if is_likely_gated:
            # Check for existing token
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            
            if not token:
                print(f"\n{'='*60}")
                print(f"⚠️  The model '{model_name}' likely requires authentication.")
                print(f"{'='*60}\n")
                print("You'll need a HuggingFace token to access this model.")
                print("\n1. Get your token from: https://huggingface.co/settings/tokens")
                print("2. Make sure you've accepted the model's terms at:")
                print(f"   https://huggingface.co/{model_name}\n")
                
                choice = input("Would you like to enter your token now? (y/n): ").lower().strip()
                
                if choice == 'y':
                    token = getpass.getpass("Enter your HuggingFace token (hf_...): ").strip()
                    if token:
                        os.environ['HF_TOKEN'] = token
                        print("✓ Token set for this session.")
                        
                        # Offer to save permanently
                        save = input("\nSave token permanently? (y/n): ").lower().strip()
                        if save == 'y':
                            self._save_token(token)
                        else:
                            print("\nTo make it permanent later, add to your shell config:")
                            print(f"export HF_TOKEN='{token[:8]}...'")
                        print(f"{'='*60}\n")
                    else:
                        print("No token entered. Continuing anyway...")
                else:
                    print("\nTo set your token later, use one of these methods:")
                    print("• export HF_TOKEN='your_token_here'")
                    print("• huggingface-cli login")
                    print(f"{'='*60}\n")
    
    def _load_model(self, model_name: str, **kwargs):
        """Load a new model, replacing the current one."""
        # Check if model might need authentication
        self._check_model_access(model_name)
        
        # Shutdown existing model if any
        if self._backend is not None:
            print(f"Unloading {self._backend.backend.model if hasattr(self._backend, 'backend') else self._backend.model}...")
            if hasattr(self._backend, 'shutdown'):
                self._backend.shutdown()
            self._backend = None
            
            # Force garbage collection to free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new model
        print(f"Loading {model_name}...")
        try:
            base_model = VLLMChatLM(model_name, **kwargs)
            self._backend = AutoBatchLM(base_model, max_batch=128, flush_ms=8)
            # Update our model name
            self.model = model_name
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            # Don't print the raw error if it's a permission error - we already have a nice message
            if isinstance(e, PermissionError):
                raise
            print(f"Failed to load {model_name}: {e}")
            raise
    
    def __call__(self, prompt: str = None, messages: List[Dict[str, str]] = None, **kwargs):
        """Call the current model."""
        if self._backend is None:
            self._initialize_default()
        return self._backend(prompt, messages, **kwargs)
    
    def forward(self, *args, **kwargs):
        """DSPy-compatible forward method."""
        if self._backend is None:
            self._initialize_default()
        return self._backend.forward(*args, **kwargs)
    
    def forward_batch(self, *args, **kwargs):
        """DSPy-compatible batch forward method."""
        if self._backend is None:
            self._initialize_default()
        return self._backend.forward_batch(*args, **kwargs)
    
    async def aforward(self, *args, **kwargs):
        """DSPy-compatible async forward method."""
        if self._backend is None:
            self._initialize_default()
        return await self._backend.aforward(*args, **kwargs)
    
    @property
    def supports_batch(self):
        """DSPy compatibility."""
        return True


# Create the global instance
llm = GlobalLLM()


def llmtogpu(model: str, temperature: float = DEFAULT_TEMPERATURE, 
             max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs):
    """
    Load a model to GPU, replacing the current model.
    
    This function unloads any existing model and loads the specified one.
    Big models can take time to load, so use only one at a time.
    
    Args:
        model: HuggingFace model ID or local path
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional vLLM parameters
    
    Examples:
        >>> llmtogpu("google/gemma-2b-it")
        >>> response = llm("Hello!")
        
        >>> llmtogpu("Qwen/Qwen2.5-1.5B-Instruct", temperature=0.0)
        >>> response = llm("Explain recursion")
    """
    global llm
    llm._load_model(model, temperature=temperature, max_tokens=max_tokens, **kwargs)


def get_gpu_memory():
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def suggest_models():
    """Suggest models based on available GPU memory."""
    memory_gb = get_gpu_memory()
    
    suggestions = []
    if memory_gb < 4:
        suggestions = [
            "Qwen/Qwen3-0.6B (0.6B params, ~1GB VRAM)",
            "Qwen/Qwen2.5-0.5B-Instruct (0.5B params, ~1GB VRAM)",
        ]
    elif memory_gb < 8:
        suggestions = [
            "Qwen/Qwen3-0.6B (0.6B params, ~1GB VRAM)",
            "Qwen/Qwen2.5-0.5B-Instruct (0.5B params, ~1GB VRAM)",
            "Qwen/Qwen2.5-1.5B-Instruct (1.5B params, ~3GB VRAM)",
            "google/gemma-2b-it (2B params, ~4GB VRAM)",
            "Qwen/Qwen2.5-3B-Instruct (3B params, ~6GB VRAM)",
        ]
    elif memory_gb < 16:
        suggestions = [
            "Qwen/Qwen2.5-3B-Instruct (3B params, ~6GB VRAM)",
            "Qwen/Qwen2.5-7B-Instruct (7B params, ~14GB VRAM)",
            "meta-llama/Llama-3.2-3B-Instruct (3B params, ~6GB VRAM)",
        ]
    else:
        suggestions = [
            "Qwen/Qwen2.5-7B-Instruct (7B params, ~14GB VRAM)",
            "Qwen/Qwen2.5-14B-Instruct (14B params, ~28GB VRAM)",
            "meta-llama/Llama-3.2-8B-Instruct (8B params, ~16GB VRAM)",
            "mistralai/Mistral-7B-Instruct-v0.3 (7B params, ~14GB VRAM)",
        ]
    
    print(f"Available GPU memory: {memory_gb:.1f} GB")
    print("Suggested models for your system:")
    for model in suggestions:
        print(f"  - {model}")
    
    return suggestions


# Help documentation
def help_ovllm():
    """
    Display comprehensive help for OVLLM.
    
    OVLLM makes running local LLMs simple and efficient with vLLM backend.
    
    Quick Start:
    -----------
    >>> from ovllm import llm
    >>> response = llm("Hello!")  # Uses default small model
    
    Switch Models:
    -------------
    >>> from ovllm import llmtogpu
    >>> llmtogpu("google/gemma-2b-it")  # Load a different model
    >>> response = llm("Explain AI")
    
    DSPy Integration:
    ----------------
    >>> import dspy, ovllm
    >>> dspy.configure(lm=ovllm.llm, adapter = dspy.TwoStepAdapter)
    >>> predict = dspy.Predict("question -> answer")
    >>> result = predict(question="What is Python?")
    
    Batch Processing with DSPy:
    --------------------------
    >>> examples = [
    ...     dspy.Example(question="What is AI?"),
    ...     dspy.Example(question="What is ML?"),
    ... ]
    >>> results = predict.batch(examples)
    
    Available Functions:
    -------------------
    - llm(prompt): Call the current model
    - llmtogpu(model): Load a new model to GPU
    - suggest_models(): Get model recommendations for your GPU
    - get_gpu_memory(): Check available GPU memory
    
    Tips:
    -----
    - Start with small models (0.5B-2B params) for testing
    - Only load one model at a time to conserve memory
    - Use temperature=0.0 for deterministic outputs
    - Batch requests with DSPy for better throughput
    
    Common Models:
    -------------
    - Qwen/Qwen2.5-0.5B-Instruct: Tiny, fast, good for testing
    - google/gemma-2b-it: Small, efficient, good quality
    - Qwen/Qwen2.5-7B-Instruct: Medium, balanced performance
    - meta-llama/Llama-3.2-3B-Instruct: Good quality, moderate size
    """
    print(help_ovllm.__doc__)


# Add help to module
llm.__doc__ = """
Global LLM instance for OVLLM.

Usage:
    >>> from ovllm import llm
    >>> response = llm("Hello!")
    
This is a singleton that manages the currently loaded model.
Use llmtogpu() to switch models.

DSPy compatible - can be used directly with dspy.configure(lm=llm)
"""

# Set module-level help
__doc__ = help_ovllm.__doc__

# Export main interface
__all__ = ['llm', 'llmtogpu', 'VLLMChatLM', 'AutoBatchLM', 
           'suggest_models', 'get_gpu_memory', 'help_ovllm']


# Print welcome message on import
print("OVLLM ready. The default model will be loaded on first use.")
print("Use llmtogpu('model-name') to load a specific model")
print("Use suggest_models() to see recommendations for your GPU")