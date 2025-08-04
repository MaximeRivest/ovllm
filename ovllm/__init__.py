"""
OVLLM – One‑line vLLM for Local Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A simple, powerful Python library to run local LLMs with the vLLM engine,
designed for ease of use and high-performance, seamless DSPy integration.

>>> import ovllm
>>> import dspy

# --- Advanced Usage & DSPy Integration ---
# Load a more powerful model, passing custom vLLM engine arguments.
>>> ovllm.llmtogpu(
...     "google/gemma-3n-E4B-it",
...     vllm_args={"tensor_parallel_size": 1} # Pass any args to vLLM
... )

# The `llm` object is a full dspy.LM. Override sampling params per-call.
>>> response_obj = ovllm.llm("Why is the sky blue?", top_p=0.9, temperature=0.5)
>>> print(response_obj.text)

# Seamlessly integrate with a DSPy program.
>>> dspy.configure(lm=ovllm.llm)
>>> predict = dspy.Predict("question -> answer")
>>> result = predict(question="What is the chemical formula for water?")
>>> print(result.answer)

# Advance params
>>> ovllm.llmtogpu(
...     "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
...     vllm_args={"tensor_parallel_size": 2, "max_model_len": 8192, "gpu_memory_utilization": 0.80})

"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. Imports & Core Dependencies
# --------------------------------------------------------------------------- #
import asyncio
import gc
import os
import threading
import warnings
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Attempt to import heavy dependencies and provide helpful errors if they fail.
try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for OVLLM. Please install it from https://pytorch.org/"
    ) from e

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError(
        "vLLM is required for OVLLM. Please install it with:\n\n"
        "   pip install vllm\n\n"
        "See https://docs.vllm.ai/en/latest/getting_started/installation.html for hardware requirements."
    ) from e

try:
    import dspy
except ImportError as e:
    raise ImportError(
        "DSPy is required for full OVLLM functionality. Please install it with:\n\n"
        "   pip install dspy-ai"
    ) from e


# --------------------------------------------------------------------------- #
# 2. Module-level Configuration & Constants
# --------------------------------------------------------------------------- #

# Public API exports
__all__ = [
    "llm",
    "llmtogpu",
    "suggest_models",
    "get_gpu_memory",
    "help_ovllm",
    "VLLMChatLM",
    "AutoBatchLM",
]
__version__ = "0.3.0" 

# A small, capable default model that can run on most systems.
_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024


# --------------------------------------------------------------------------- #
# 3. Core LM Implementation (Internal Classes)
# --------------------------------------------------------------------------- #
import time
from types import SimpleNamespace 

def _wrap_request_output(vllm_output, model_name: str) -> SimpleNamespace:
    """Convert a vLLM output into an OpenAI‑compatible object for DSPy."""
    comp = vllm_output.outputs[0]

    # assistant message
    message = SimpleNamespace(role="assistant", content=comp.text)

    # one choice
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=getattr(comp, "finish_reason", "stop"),
    )

    # usage statistics
    usage = {
        "prompt_tokens": len(vllm_output.prompt_token_ids),
        "completion_tokens": len(comp.token_ids),
        "total_tokens": len(vllm_output.prompt_token_ids) + len(comp.token_ids),
    }

    # full response object
    resp = SimpleNamespace(
        id=f"ovllm-{time.time_ns()}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=usage,
    )

    # Convenience field so existing OVLLM examples (`resp.text`) keep working
    resp.text = comp.text
    return resp

class VLLMChatLM(dspy.BaseLM):
    """A thin wrapper around a vLLM engine that speaks the DSPy BaseLM protocol."""
    supports_batch = True
    def __init__(
        self,
        model: str,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        vllm_args: Optional[Dict[str, Any]] = None,
        **sampler_overrides,
    ) -> None:
        super().__init__(model=model)
        self.provider = "vllm"
        self.model_type = "chat"
        
        # Consolidate sampling parameters, allowing kwargs to override defaults.
        self._base_sampling = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **sampler_overrides,
        }
        
        # The vLLM engine is the heaviest object, so we initialize it last.
        self._engine = None
        
        # Prepare arguments for the vLLM engine, allowing user overrides.
        engine_args = {
            "model": model,
            "dtype": "auto",
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.80,
            "max_model_len": 8192
        }
        if torch.cuda.is_available():
            engine_args["tensor_parallel_size"] = torch.cuda.device_count()
        
        if vllm_args:
            engine_args.update(vllm_args)

        try:
            self._engine = LLM(**engine_args)
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                raise MemoryError(
                    f"❌ Not enough GPU memory to load '{model}' with settings {engine_args}.\n"
                    f"   Try a smaller model or enable quantization (e.g., `vllm_args={{'quantization': 'awq'}}`).\n"
                    f"   Call `ovllm.suggest_models()` for recommendations."
                ) from exc
            if "401" in msg or "gated" in msg:
                raise PermissionError(
                    f"❌ The HuggingFace repository for '{model}' is gated.\n"
                    f"   You need to grant access on HuggingFace and provide a token.\n"
                    f"   1. Visit https://huggingface.co/settings/tokens to create a token.\n"
                    f"   2. Run `huggingface-cli login` in your terminal and paste the token."
                ) from exc
            raise

    def forward(self, prompt: str | None = None, messages: List[Dict[str, str]] | None = None, **kw) -> List[Dict]:
        return self.forward_batch([prompt], [messages], **kw)

    async def aforward(self, prompt: str | None = None, messages: List[Dict[str, str]] | None = None, **kw) -> List[Dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.forward(prompt, messages, **kw))

    def forward_batch(
        self,
        prompts: Sequence[str | None],
        messages_list: Sequence[List[Dict[str, str]] | None] | None = None,
        **kw,
    ) -> List[Dict]:
        if messages_list is None:
            messages_list = [None] * len(prompts)
        
        tokenizer = self._engine.get_tokenizer()
        
        formatted_prompts = []
        for p, m in zip(prompts, messages_list):
            if m is None:
                m = [{"role": "user", "content": p or ""}]
            
            # This check is now robust for both message lists and raw prompts
            if isinstance(m, list) and all(isinstance(i, dict) for i in m):
                formatted_prompts.append(
                    tokenizer.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                )
            else:
                 # Fallback for raw prompts
                formatted_prompts.append(p or "")

        # Per-call sampling params (`kw`) override the instance's base sampling params.
        sampling_params = SamplingParams(**{**self._base_sampling, **kw})
        raw_outputs = self._engine.generate(formatted_prompts, sampling_params, use_tqdm=False)
        return [_wrap_request_output(o, self.model) for o in raw_outputs]

    def shutdown(self) -> None:
        """Gracefully shuts down the vLLM engine and releases GPU memory."""
        if self._engine is not None:
            # vLLM's internal shutdown logic can be complex; this is a best-effort attempt.
            if hasattr(self._engine, "llm_engine") and hasattr(self._engine.llm_engine, "destroy"):
                 self._engine.llm_engine.destroy()
            self._engine = None

class AutoBatchLM(dspy.BaseLM):
    """A micro-batching wrapper for any DSPy LM to maximize GPU throughput."""
    supports_batch = True

    def __init__(self, backend: VLLMChatLM, *, max_batch: int = 128, flush_ms: int = 10):
        super().__init__(model=backend.model)
        self.backend = backend
        self.max_batch = max_batch
        self.flush_ms = flush_ms
        
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def forward(self, prompt=None, messages=None, **kw):
        fut = asyncio.run_coroutine_threadsafe(
            self._enqueue(prompt, messages, kw), self._loop
        )
        return fut.result()          # ← return the object itself

    async def aforward(self, prompt=None, messages=None, **kw):
        loop = asyncio.get_running_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._enqueue(prompt, messages, kw), self._loop
        )
        return await asyncio.wrap_future(fut, loop=loop)
    
    def forward_batch(self, prompts, messages_list=None, **kw):
        """For explicit batching, bypass the queue and call the backend directly."""
        return self.backend.forward_batch(prompts, messages_list, **kw)

    async def _enqueue(self, p, m, kw):
        fut = self._loop.create_future()
        await self._queue.put((p, m, kw, fut))
        return await fut

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready.set()
        self._loop.create_task(self._collector())
        self._loop.run_forever()

    async def _collector(self):
        from asyncio import QueueEmpty
        while self._loop.is_running():
            p, m, kw, fut = await self._queue.get()
            bucket = [(p, m, kw, fut)]
            t0 = self._loop.time()
            while len(bucket) < self.max_batch:
                try:
                    bucket.append(self._queue.get_nowait())
                except QueueEmpty:
                    break
            while len(bucket) < self.max_batch and (self._loop.time() - t0) * 1000 < self.flush_ms:
                try:
                    bucket.append(await asyncio.wait_for(self._queue.get(), timeout=0.001))
                except asyncio.TimeoutError:
                    break
            by_kw: Dict[Tuple[Tuple[str, Any], ...], List[Tuple]] = defaultdict(list)
            for p_i, m_i, kw_i, fut_i in bucket:
                by_kw[tuple(sorted(kw_i.items()))].append((p_i, m_i, fut_i))
            for kw_key, group in by_kw.items():
                prompts_list = [item[0] for item in group]
                msgs_list = [item[1] for item in group]
                futures = [item[2] for item in group]
                shared_kw = dict(kw_key)
                try:
                    outputs = self.backend.forward_batch(prompts_list, msgs_list, **shared_kw)
                    if len(outputs) != len(group):
                        raise RuntimeError(f"Batch processing failed: expected {len(group)} outputs, got {len(outputs)}")
                    for out, f in zip(outputs, futures):
                        if not f.done(): f.set_result(out)
                except Exception as exc:
                    for f in futures:
                        if not f.done(): f.set_exception(exc)

    def shutdown(self):
        """Shuts down the backend engine and the asyncio processing loop."""
        if hasattr(self.backend, "shutdown"):
            self.backend.shutdown()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# 4. Global Singleton Management
# --------------------------------------------------------------------------- #

_GLOBAL_LM_LOCK = threading.RLock()
_GLOBAL_LM_INSTANCE: Optional[AutoBatchLM] = None


def llmtogpu(
    model: str,
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    vllm_args: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Loads a new Language Model onto the GPU, making it the active model for `ovllm.llm`.

    This function handles the complete lifecycle: if a model is already loaded,
    it is gracefully unloaded and its memory is freed before the new one is loaded.

    Args:
        model (str): The model identifier from HuggingFace Hub (e.g., "google/gemma-2b-it").
        temperature (float): The default sampling temperature for generation (0.0 for deterministic).
        max_tokens (int): The default maximum number of tokens to generate.
        vllm_args (Dict[str, Any], optional): A dictionary of advanced arguments to pass
            directly to the vLLM engine constructor. Use this for settings like `quantization`,
            `dtype`, `swap_space`, `enforce_eager`, etc.
        **kwargs: Additional default sampling parameters (e.g., `top_p`, `top_k`).
    """
    global _GLOBAL_LM_INSTANCE
    with _GLOBAL_LM_LOCK:
        if _GLOBAL_LM_INSTANCE is not None:
            print(f"INFO: Unloading previous model: {_GLOBAL_LM_INSTANCE.backend.model}")
            _GLOBAL_LM_INSTANCE.shutdown()
            _GLOBAL_LM_INSTANCE = None
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"INFO: Loading new model: {model}. Please wait...")
        try:
            vllm_backend = VLLMChatLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                vllm_args=vllm_args,
                **kwargs
            )
            _GLOBAL_LM_INSTANCE = AutoBatchLM(vllm_backend)
            llm.model = model # Update the public proxy object's model name
            print(f"✅ SUCCESS: Model '{model}' is loaded and ready.")
        except (MemoryError, PermissionError, Exception) as e:
            # The VLLMChatLM init will raise specific, user-friendly errors.
            print(f"\n{e}")
            _GLOBAL_LM_INSTANCE = None

class _LLMProxy(dspy.BaseLM):
    """
    A lightweight, copy-safe proxy for the global LLM instance.
    This is the object users interact with as `ovllm.llm`. It delegates all calls
    to the single, active `_GLOBAL_LM_INSTANCE` and handles lazy initialization.
    """
    def __init__(self, default_model: str):
        super().__init__(model=default_model)

    def _ensure_initialized(self):
        """Initializes the default model if no model is currently loaded."""
        if _GLOBAL_LM_INSTANCE is None:
            with _GLOBAL_LM_LOCK:
                # Double-check lock to prevent race conditions
                if _GLOBAL_LM_INSTANCE is None:
                    print(f"INFO: No model loaded. Initializing with default: {_DEFAULT_MODEL}")
                    llmtogpu(_DEFAULT_MODEL)
                    if _GLOBAL_LM_INSTANCE is None:
                        raise RuntimeError(
                            "Failed to initialize the default model. "
                            "Please check your vLLM and CUDA setup."
                        )

    def __call__(self, *args, **kwargs):
        self._ensure_initialized()
        # This executes BaseLM.__call__, which logs history.
        return _GLOBAL_LM_INSTANCE(*args, **kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access (e.g., .forward) to the active instance."""
        with _GLOBAL_LM_LOCK:
            self._ensure_initialized()
        # This will correctly fetch .forward, .aforward, etc. from the active AutoBatchLM instance
        return getattr(_GLOBAL_LM_INSTANCE, name)

    def forward(self, *args, **kwargs):
        self._ensure_initialized()
        return _GLOBAL_LM_INSTANCE.forward(*args, **kwargs)

    async def aforward(self, *args, **kwargs):
        self._ensure_initialized()
        return await _GLOBAL_LM_INSTANCE.aforward(*args, **kwargs)
    
    def inspect_history(self, n: int = 1):
        if _GLOBAL_LM_INSTANCE is None:
            print("No model currently loaded.")
            return []
        # Re‑use DSPy’s pretty printer on the real LM instance
        return _GLOBAL_LM_INSTANCE.inspect_history(n)



# --------------------------------------------------------------------------- #
# 5. Public API Instantiation & Helper Functions
# --------------------------------------------------------------------------- #

llm = _LLMProxy(default_model=_DEFAULT_MODEL)

import subprocess
import re

def get_gpu_memory() -> tuple[float, int]:
    """
    Attempt to detect the total GPU memory (in GB) and the number of GPUs.

    Returns
    -------
    tuple
        A tuple ``(vram_gb, gpu_count)`` describing the total available VRAM
        across all GPUs in the system.  If no GPU is detected, returns
        (0.0, 0).
    """
    try:
        # Try to query NVIDIA GPUs via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Parse the output (one line per GPU)
        mems = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if not mems:
            return 0.0, 0
        return sum(mems) / 1024.0, len(mems)  # convert MiB to GiB
    except (FileNotFoundError, subprocess.CalledProcessError):
        # If nvidia-smi is not available, try to detect via other tools
        return 0.0, 0


def suggest_models() -> None:
    """
    Print a list of suggested Hugging Face models tailored to the available
    GPU memory on the current system.

    The recommendations are organised into VRAM tiers.  For each tier we list
    instruction‑tuned or chat‑optimised checkpoints that fit comfortably in
    the given memory budget.  These suggestions are based on publicly
    available documentation: small models like Qwen3‐0.6B (≈0.6 B parameters)
    and Gemma 3 1B require only around a gigabyte of GPU memory【981739344745879†L89-L98】【154918926046664†L68-L120】,
    while larger mixtures‑of‑experts and 24–30 B parameter models can
    saturate a 32 GB card【554109789521618†L64-L73】【37215507856503†L692-L721】.
    """
    vram, gpu_count = get_gpu_memory()
    print(f"Detected {vram:.1f} GB of VRAM across {gpu_count} GPU(s).\n")

    if vram == 0.0:
        print("No compatible GPU detected. vLLM performance may be limited.")
        return

    print("Suggested models for your system:")
    # sub‑4 GB VRAM: tiny instruction‑tuned models
    if vram < 4:
        # Qwen3‑0.6B has 0.6 billion parameters【981739344745879†L89-L98】.
        print("  - Qwen/Qwen3-0.6B-Instruct (≈0.6 B params, ~1 GB VRAM)")
        # Gemma 3 1B is the smallest member of the Gemma 3 family【154918926046664†L68-L120】.
        print("  - google/gemma-3-1b-it (1 B params, ~2 GB VRAM)")
    # 4–7 GB VRAM: small to medium models
    elif vram < 8:
        print("  - Qwen/Qwen3-0.6B-Instruct (≈0.6 B params, ~1 GB VRAM)")
        # Qwen3‑4B has around 4 billion parameters【563728763966739†L92-L100】.
        print("  - Qwen/Qwen3-4B-Instruct (4 B params, ~6 GB VRAM)")
        # Gemma 3 4B is a 4 billion parameter model in the Gemma 3 lineup【154918926046664†L68-L120】.
        print("  - google/gemma-3-4b-it (4 B params, ~6 GB VRAM)")
    # 8–15 GB VRAM: medium‑sized chat models
    elif vram < 16:
        # Gemma 3 12B is a mid‑sized 12 billion parameter model【154918926046664†L68-L120】.
        print("  - google/gemma-3-12b-it (12 B params, ~12 GB VRAM)")
        # Meta Llama 3 8B is an 8 billion parameter model; GGUF quantised versions fit in ~10 GB.
        print("  - meta-llama/Llama-3-8B-Instruct (8 B params, ~10 GB VRAM)")
        # Mistral 7B Instruct is a 7 billion parameter chat model.
        print("  - mistralai/Mistral-7B-Instruct-v0.2 (7 B params, ~12 GB VRAM)")
    # 16–31 GB VRAM: large single‑GPU models
    elif vram < 32:
        # Mistral Small 3.1 has 24 billion parameters and fits on a 32 GB card【554109789521618†L64-L73】.
        print("  - mistralai/Mistral-Small-3.1-24B-Instruct-2503 (24 B params, fits in 32 GB VRAM)")
        # Qwen3‑30B‑A3B is a 30 billion parameter mixture‑of‑experts model that fits in 30 GB【37215507856503†L692-L721】.
        print("  - Qwen/Qwen3-30B-A3B-Instruct-2507 (30 B params, fits in 30 GB VRAM)")
        # Gemma 3 27B is the largest Gemma 3 variant【154918926046664†L68-L120】.
        print("  - google/gemma-3-27b-it (27 B params, ~28–32 GB VRAM)")
    # ≥32 GB VRAM: ultra‑large and Mixture‑of‑Experts models
    else:
        # Kimi K2 is a mixture‑of‑experts model with 32 billion active parameters【991836247192872†L94-L124】.
        print("  - moonshotai/Kimi-K2-Instruct (1 T total / 32 B active params)")
        # GLM‑4.5‑Air has 12 billion active parameters【564556693604147†L60-L69】.
        print("  - zai-org/GLM-4.5-Air (12 B active params)")
        # GLM‑4.5 has 32 billion active parameters【564556693604147†L60-L69】.
        print("  - zai-org/GLM-4.5 (32 B active params)")
        print("  - With ≥32 GB VRAM you may also explore experimental models up to 70 B parameters.")


def help_ovllm():
    """Prints the main library documentation."""
    print(__doc__)

# Print a friendly welcome message on import
print("OVLLM Initialized. Use `help_ovllm()` for a guide, or `suggest_models()` for recommendations.")