"""Basic tests for OVLLM functionality."""

import pytest
import ovllm


def test_import():
    """Test that the package can be imported."""
    assert hasattr(ovllm, 'llm')
    assert hasattr(ovllm, 'llmtogpu')
    assert hasattr(ovllm, 'suggest_models')
    assert hasattr(ovllm, 'get_gpu_memory')


def test_llm_callable():
    """Test that llm is callable."""
    assert callable(ovllm.llm)


def test_dspy_compatibility():
    """Test that llm has DSPy-compatible attributes."""
    assert hasattr(ovllm.llm, 'forward')
    assert hasattr(ovllm.llm, 'forward_batch')
    assert hasattr(ovllm.llm, 'aforward')
    assert hasattr(ovllm.llm, 'supports_batch')
    assert ovllm.llm.supports_batch is True


def test_gpu_memory():
    """Test GPU memory detection."""
    memory = ovllm.get_gpu_memory()
    assert isinstance(memory, (int, float))
    assert memory >= 0


def test_suggest_models():
    """Test model suggestions."""
    suggestions = ovllm.suggest_models()
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0


def test_help_function():
    """Test that help function exists and is callable."""
    assert hasattr(ovllm, 'help_ovllm')
    assert callable(ovllm.help_ovllm)


if __name__ == "__main__":
    # Run basic import test
    print("Testing OVLLM basic functionality...")
    test_import()
    print("✓ Import test passed")
    
    test_llm_callable()
    print("✓ LLM callable test passed")
    
    test_dspy_compatibility()
    print("✓ DSPy compatibility test passed")
    
    test_gpu_memory()
    print("✓ GPU memory test passed")
    
    test_suggest_models()
    print("✓ Model suggestions test passed")
    
    test_help_function()
    print("✓ Help function test passed")
    
    print("\nAll basic tests passed!")