#!/usr/bin/env python3
"""
Example demonstrating DSPy integration with OVLLM.
"""

import dspy
import ovllm


def basic_example():
    """Basic DSPy usage with OVLLM"""
    print("=== Basic DSPy Example ===\n")
    
    # Configure DSPy to use OVLLM
    dspy.configure(lm=ovllm.llm)
    
    # Create a simple prediction
    predict = dspy.Predict("question -> answer")
    result = predict(question="What is Python?")
    print(f"Question: What is Python?")
    print(f"Answer: {result.answer}")
    print()


def chain_of_thought_example():
    """Chain of Thought reasoning with DSPy"""
    print("=== Chain of Thought Example ===\n")
    
    # Define a signature for mathematical reasoning
    class MathSignature(dspy.Signature):
        """Solve math problems step by step."""
        problem = dspy.InputField()
        reasoning = dspy.OutputField(desc="Step-by-step reasoning")
        answer = dspy.OutputField(desc="Final numerical answer")
    
    # Create a Chain of Thought predictor
    cot = dspy.ChainOfThought(MathSignature)
    
    result = cot(problem="If I have 5 apples and buy 3 more, then give away 2, how many do I have?")
    print(f"Problem: {result.problem}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")
    print()


def batch_processing_example():
    """Batch processing with automatic batching"""
    print("=== Batch Processing Example ===\n")
    
    # Create examples for batch processing
    examples = [
        dspy.Example(question="What is artificial intelligence?").with_inputs("question"),
        dspy.Example(question="What is machine learning?").with_inputs("question"),
        dspy.Example(question="What is deep learning?").with_inputs("question"),
        dspy.Example(question="What is neural network?").with_inputs("question"),
    ]
    
    # Create predictor
    predict = dspy.Predict("question -> answer")
    
    # Process batch - OVLLM will automatically batch these
    print("Processing batch of questions...")
    results = predict.batch(examples)
    
    for i, (example, result) in enumerate(zip(examples, results)):
        print(f"{i+1}. Q: {example.question}")
        print(f"   A: {result.answer[:100]}...")  # Truncate for display
        print()


def rag_example():
    """RAG-style example with context"""
    print("=== RAG Example ===\n")
    
    # Define a RAG signature
    class RAGSignature(dspy.Signature):
        """Answer questions based on provided context."""
        context = dspy.InputField(desc="Relevant information")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="Answer based on context")
    
    # Create predictor
    rag = dspy.ChainOfThought(RAGSignature)
    
    # Example with context
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
    Constructed from 1887 to 1889, it was initially criticized by some of France's leading 
    artists and intellectuals for its design, but it has become a global cultural icon of 
    France and one of the most recognizable structures in the world. The Eiffel Tower is 
    the most-visited paid monument in the world; 6.91 million people ascended it in 2015.
    """
    
    result = rag(
        context=context,
        question="When was the Eiffel Tower built?"
    )
    
    print(f"Context: {context[:100]}...")
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print()


def model_switching_example():
    """Example showing model switching"""
    print("=== Model Switching Example ===\n")
    
    # Show current model
    print(f"Current model: {ovllm.llm.model}")
    
    # List available models
    print("\nAvailable models for your GPU:")
    ovllm.suggest_models()
    
    print("\nTo switch models, use:")
    print('>>> ovllm.llmtogpu("Qwen/Qwen2.5-1.5B-Instruct")')
    print()


def main():
    """Run all examples"""
    print("OVLLM DSPy Integration Examples")
    print("=" * 50)
    print()
    
    # Run examples
    basic_example()
    chain_of_thought_example()
    batch_processing_example()
    rag_example()
    model_switching_example()
    
    print("=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()