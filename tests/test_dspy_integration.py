"""Tests for DSPy integration with OVLLM."""

import pytest
import dspy
import ovllm
from types import SimpleNamespace


def test_dspy_forward_output_format():
    """Test that VLLMChatLM returns correct format for DSPy."""
    # Create a minimal VLLMChatLM instance
    # This tests the output format without loading a real model
    
    # Mock output to test format
    mock_output = SimpleNamespace(
        outputs=[SimpleNamespace(
            text="Test response",
            finish_reason="stop",
            token_ids=[1, 2, 3]
        )],
        prompt_token_ids=[4, 5, 6]
    )
    
    # Test the wrapper function
    result = ovllm._wrap_request_output(mock_output, "test-model")
    
    # Check structure matches DSPy expectations
    assert hasattr(result, 'model')
    assert hasattr(result, 'choices')
    assert hasattr(result, 'usage')
    
    assert result.model == "test-model"
    assert len(result.choices) == 1
    assert hasattr(result.choices[0], 'message')
    assert hasattr(result.choices[0].message, 'content')
    assert result.choices[0].message.content == "Test response"
    
    assert 'prompt_tokens' in result.usage
    assert 'completion_tokens' in result.usage
    assert 'total_tokens' in result.usage


def test_dspy_base_lm_interface():
    """Test that our classes properly implement DSPy's BaseLM interface."""
    # Test VLLMChatLM inheritance
    assert issubclass(ovllm.VLLMChatLM, dspy.BaseLM)
    
    # Test AutoBatchLM inheritance
    assert issubclass(ovllm.AutoBatchLM, dspy.BaseLM)
    
    # Test GlobalLLM inheritance
    assert isinstance(ovllm.llm, dspy.BaseLM)


def test_dspy_required_methods():
    """Test that all required DSPy methods are present."""
    required_methods = ['forward', 'forward_batch', 'aforward']
    
    for method in required_methods:
        assert hasattr(ovllm.llm, method)
        assert callable(getattr(ovllm.llm, method))


def test_dspy_supports_batch():
    """Test that batch support is properly declared."""
    assert hasattr(ovllm.llm, 'supports_batch')
    assert ovllm.llm.supports_batch is True


def test_autobatch_wrapper():
    """Test AutoBatchLM wrapper functionality."""
    # Create a mock backend
    mock_backend = SimpleNamespace(
        model="test-model",
        model_type="chat",
        temperature=0.7,
        max_tokens=512,
        forward_batch=lambda prompts, messages_list, **kw: [
            SimpleNamespace(
                model="test-model",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=f"Response to: {p}")
                )],
                usage={}
            ) for p in prompts
        ]
    )
    
    # Wrap with AutoBatchLM
    batched = ovllm.AutoBatchLM(mock_backend, max_batch=10, flush_ms=1)
    
    # Test properties
    assert batched.model == "test-model"
    assert batched.supports_batch is True
    
    # Cleanup
    batched.shutdown()


def test_dspy_signature_compatibility():
    """Test that our implementation works with DSPy signatures."""
    # This tests the signature parsing without actual model calls
    
    # Test simple signature
    sig = dspy.Signature("question -> answer")
    assert 'question' in sig.input_fields
    assert 'answer' in sig.output_fields
    
    # Test custom signature
    class TestSignature(dspy.Signature):
        """Test signature for OVLLM."""
        input_text = dspy.InputField()
        output_text = dspy.OutputField(desc="Generated output")
    
    sig = TestSignature()
    assert 'input_text' in sig.input_fields
    assert 'output_text' in sig.output_fields


if __name__ == "__main__":
    print("Testing OVLLM DSPy integration...")
    
    test_dspy_forward_output_format()
    print("✓ Forward output format test passed")
    
    test_dspy_base_lm_interface()
    print("✓ BaseLM interface test passed")
    
    test_dspy_required_methods()
    print("✓ Required methods test passed")
    
    test_dspy_supports_batch()
    print("✓ Batch support test passed")
    
    test_autobatch_wrapper()
    print("✓ AutoBatch wrapper test passed")
    
    test_dspy_signature_compatibility()
    print("✓ Signature compatibility test passed")
    
    print("\nAll DSPy integration tests passed!")