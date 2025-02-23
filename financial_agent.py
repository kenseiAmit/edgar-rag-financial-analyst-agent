#!/usr/bin/env python3
"""
Financial Agent - Main System Interface
Agentic EDGAR-RAG with Multi-Tool Coordination

This script provides the end-to-end user-facing entry point for the Financial Agent.
It coordinates 6 specialized tools across document analysis, relational database queries,
real-time market data retrieval, and automatic PII privacy masking.
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure local modules are accessible
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Load environment configuration
load_dotenv()

from helper_modules.agent_coordinator import AgentCoordinator


def print_banner():
    banner = """
================================================================================
🏦 FINTOOL ANALYST: AGENTIC MULTI-TOOL FINANCIAL INTELLIGENCE SYSTEM
================================================================================
Tools Ecosystem:
  • 📄 Document RAG: AAPL, GOOGL, TSLA 10-K SEC Filings (LlamaIndex Vector Store)
  • 💾 Database Query: Natural Language to SQLite SQL (Customer & Portfolio Data)
  • 📈 Market Search: Real-time Stock Pricing & Volume (Yahoo Finance API)
  • 🔒 Privacy Guard: Automated PII Detection & Context-Preserving Masking
================================================================================
"""
    print(banner)


def run_benchmark_queries(agent: AgentCoordinator):
    """Runs a series of diverse benchmark queries demonstrating multi-tool coordination."""
    benchmark_queries = [
        ("Document RAG", "What are Apple's main business segments and revenue drivers?"),
        ("Database SQL", "How many customers are in the database grouped by investment profile?"),
        ("Real-time Market Data", "What is Tesla's current stock price and trading volume?"),
        ("Automatic PII Protection", "Show customer names, emails, and portfolio values for Apple stock holders"),
        ("Dual-Tool Coordination (DB + Market)", "Compare current Tesla stock price with what our Tesla-holding customers paid based on portfolio data"),
        ("Triple-Tool Coordination (Doc + DB + Market)", "Analyze Apple comprehensively: show business segments from 10-K, current stock price, and customers who hold Apple stock")
    ]

    print("\n🚀 Executing Benchmark Test Queries across all capabilities...\n")
    for category, query_str in benchmark_queries:
        print(f"\n{'=' * 80}")
        print(f"📌 Scenario: {category}")
        print(f"❓ Query: {query_str}")
        print(f"{'-' * 80}")
        try:
            response = agent.query(query_str, verbose=True)
            print(f"\n💡 Agent Response:\n{response}")
        except Exception as e:
            print(f"❌ Error executing query: {e}")
        print(f"{'=' * 80}\n")


def interactive_mode(agent: AgentCoordinator):
    """Interactive query loop for user input."""
    print("\n💬 Entering Interactive Mode. Type your question or 'exit' / 'quit' to end.\n")
    while True:
        try:
            user_input = input("FinTool Analyst > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Exiting FinTool Analyst. Goodbye!")
                break
            
            response = agent.query(user_input, verbose=True)
            print(f"\n💡 Response:\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="FinTool Analyst - Agentic EDGAR-RAG with Multi-Tool Coordination"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process through the agent coordinator"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run comprehensive benchmark queries across all tools"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Display status and available tools of the agent coordinator"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose coordination output"
    )

    args = parser.parse_args()

    print_banner()

    # Initialize the coordinator
    verbose = not args.quiet
    agent = AgentCoordinator(verbose=verbose)
    agent.setup()

    if args.status:
        status = agent.get_status()
        print("\n📊 Agent Coordinator Status:")
        for k, v in status.items():
            print(f"  • {k}: {v}")
        
        tools = agent.get_available_tools()
        print(f"\n🛠️ Available Tools ({tools['total_tools']} Total):")
        print("  Document Tools:")
        for t in tools['document_tools']:
            print(f"    - {t}")
        print("  Function Tools:")
        for t in tools['function_tools']:
            print(f"    - {t}")
        return

    if args.query:
        print(f"\n❓ Processing Query: {args.query}\n")
        response = agent.query(args.query, verbose=verbose)
        print(f"\n💡 Final Response:\n{response}")
    elif args.benchmark:
        run_benchmark_queries(agent)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
