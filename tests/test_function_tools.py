"""
Test Function Tools Module - Educational Testing Framework

This test file helps you verify your FunctionToolsManager implementation step by step.
Run this file to check your progress and get helpful debugging hints.

Usage: python tests/test_function_tools.py

The tests are designed to:
1. Check each function tool individually
2. Provide specific error messages and hints
3. Help you understand what went wrong and how to fix it
4. Build confidence as you progress through the implementation

Test Categories:
- Configuration Tests: Verify LLM setup and database connection
- Database Tool Tests: Check SQL generation and execution
- Market Data Tool Tests: Verify API integration and data fetching
- PII Protection Tool Tests: Check privacy protection functionality
- Integration Tests: End-to-end tool functionality verification
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from helper_modules.function_tools import FunctionToolsManager
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.tools import FunctionTool
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Hint: Make sure you're running this from the project root directory")
    print("💡 Hint: Check that all required packages are installed: pip install -r requirements.txt")
    sys.exit(1)
