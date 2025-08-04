# Changelog

All notable changes to OVLLM will be documented in this file.

## [Unreleased]

### Fixed
- Fixed DSPy integration by implementing proper output formatting in `VLLMChatLM.forward_batch()`
- The `_wrap_request_output` function now correctly converts vLLM outputs to OpenAI-style format expected by DSPy
- AutoBatchLM now properly inherits from `dspy.BaseLM` and implements all required methods
- GlobalLLM singleton now correctly delegates DSPy methods to the backend

### Added
- Comprehensive DSPy integration examples in `examples/dspy_integration.py`
- DSPy-specific tests in `tests/test_dspy_integration.py`
- Chain of Thought reasoning examples in README
- Improved documentation for DSPy integration

### Changed
- Updated README with more comprehensive DSPy examples
- Improved error messages for model loading failures

## [0.1.2] - Previous Release

Initial public release with basic vLLM wrapper functionality.