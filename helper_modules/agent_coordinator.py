"""
Agent Coordinator Module - Complete Financial Agent with Modular Architecture

This module provides the complete financial agent functionality with intelligent routing,
tool coordination, and backward compatibility. It provides all functionality in a single
coordinated system using modular architecture.

Learning Objectives:
- Understand multi-tool coordination and intelligent routing
- Implement LLM-based decision making for tool selection
- Learn result synthesis from multiple data sources
- Build modular agent architecture
- Master PII protection in agent workflows

Key Features:
- Multi-tool coordination with intelligent routing
- Document analysis (10-K filings) for Apple, Google, Tesla
- Database queries with SQL auto-generation and PII protection
- Real-time market data from Yahoo Finance
- Complete backward compatibility for existing notebooks
- Modular architecture using helper modules
"""

import os
import ast
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Complete Financial Agent with Dynamic Multi-Tool Coordination
    
    This class combines document tools and function tools into a single
    coordinated system using modular architecture.
    
    Architecture:
    - Document Tools (3): Individual SEC 10-K filing analysis for Apple, Google, Tesla
    - Function Tools (3): Database SQL queries, real-time market data, PII protection
    - Intelligent Routing: LLM-based tool selection and result synthesis
    - Backward Compatibility: Works with existing notebooks and test suites
    """
    
    # About 4 characters per token: three sources at 8,000 characters each,
    # plus the prompt, stay well inside gpt-3.5-turbo's 16,385-token window.
    MAX_SOURCE_CHARS = 8000

    def _fit_for_synthesis(self, content: str) -> str:
        """Shorten one tool result, keeping its head and its tail.

        The tail holds the COLUMNS line and the PII notice, so both survive.
        """
        if len(content) <= self.MAX_SOURCE_CHARS:
            return content
        head = content[: self.MAX_SOURCE_CHARS - 1000]
        tail = content[-1000:]
        return f"{head}\n[... middle of this result trimmed to fit ...]\n{tail}"

    def __init__(self, companies: List[str] = None, verbose: bool = False):
        """
        Initialize the complete financial agent with modular architecture.
        
        Args:
            companies: List of company symbols (default: ["AAPL", "GOOGL", "TSLA"])
            verbose: Whether to show detailed operation information
        """
        self.companies = companies if companies is not None else ["AAPL", "GOOGL", "TSLA"]
        self.verbose = verbose
        self.project_root = Path.cwd()
        
        # Company metadata
        self.company_info = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"}
        }
        
        # Storage for tools and managers
        self.document_tools = []
        self.function_tools = []
        self.llm = None
        self.embed_model = None
        self.doc_manager = None
        self.func_manager = None
        
        self._configure_settings()
        
        # Don't auto-initialize tools - create them lazily when first needed
        self._tools_initialized = False
        
        if self.verbose:
            print("✅ Financial Agent Coordinator Initialized")
            print(f"   Companies: {self.companies}")
            print("   Tools will be created automatically when first query is made")
    
    def _configure_settings(self):
        """Configure LlamaIndex settings with OpenAI API compatibility
        
        Requirements:
        - Create OpenAI LLM with "gpt-3.5-turbo" model and temperature=0
        - Create OpenAIEmbedding with "text-embedding-ada-002" model
        - Use api_base parameter for OpenAI API compatibility
        - Set Settings.llm and Settings.embed_model
        - Store LLM reference in self.llm for routing decisions
        """
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_base=api_base,
            api_key=api_key
        )
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_base=api_base,
            api_key=api_key
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
    
    def setup(self, document_tools: List = None, function_tools: List = None):
        """
        Setup all components using the modular architecture.
        
        Args:
            document_tools: Optional pre-created document tools
            function_tools: Optional pre-created function tools
        """
        if self.verbose:
            print("🔧 Setting up Advanced Financial Agent (Modular Architecture)...")
        
        try:
            if document_tools is not None and function_tools is not None:
                self.document_tools = document_tools
                self.function_tools = function_tools
            else:
                self._create_tools()
            
            self._tools_initialized = True
            
            if self.verbose:
                status = self.get_status()
                print(f"✅ Setup complete: {status['document_tools']} document tools, {status['function_tools']} function tools")
                print(f"🎯 System ready: {'✅' if status['ready'] else '❌'}")
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if self.verbose:
                print(f"❌ Setup failed: {e}")

    def _create_tools(self):
        """Create all tools automatically using helper modules"""
        try:
            from .document_tools import DocumentToolsManager
            from .function_tools import FunctionToolsManager
        except ImportError:
            try:
                from helper_modules.document_tools import DocumentToolsManager
                from helper_modules.function_tools import FunctionToolsManager
            except ImportError:
                from document_tools import DocumentToolsManager
                from function_tools import FunctionToolsManager

        self.doc_manager = DocumentToolsManager(verbose=self.verbose)
        self.document_tools = self.doc_manager.build_document_tools()
        
        self.func_manager = FunctionToolsManager(verbose=self.verbose)
        self.function_tools = self.func_manager.create_function_tools()

    def _detect_pii_fields(self, field_names: list) -> set:
        """Detect which fields contain PII based on field names
        
        Args:
            field_names: List of database column names
            
        Returns:
            Set of field names that contain PII
        """
        pii_patterns = {
            'email', 'email_address', 'phone', 'phone_number',
            'first_name', 'last_name', 'name', 'customer_name',
            'address', 'ssn', 'social_security'
        }
        detected = set()
        for field in field_names:
            f_norm = str(field).strip().lower()
            if any(pattern in f_norm for pattern in pii_patterns):
                detected.add(str(field))
        return detected

    def _check_and_apply_pii_protection(self, tool_name: str, result: str) -> str:
        """Check if database results need PII protection and apply it automatically
        
        Args:
            tool_name: Name of the tool that generated the result
            result: Raw result string from the tool
            
        Returns:
            Protected result string with PII masked if necessary
        """
        if "database_query_tool" not in tool_name:
            return result
        
        if "COLUMNS:" not in result:
            return result
        
        # Extract column names from result
        columns = []
        match = re.search(r'COLUMNS:\s*(\[[^\]]+\])', result)
        if match:
            try:
                val = ast.literal_eval(match.group(1))
                if isinstance(val, list):
                    columns = val
            except Exception:
                columns = [c.strip().strip("'\"[]") for c in match.group(1).split(',')]
        
        pii_fields = self._detect_pii_fields(columns)
        if not pii_fields:
            return result
        
        # Invoke PII protection tool
        pii_tool = None
        for tool in self.function_tools:
            if 'pii' in getattr(tool.metadata, 'name', ''):
                pii_tool = tool
                break
        
        if pii_tool is not None:
            try:
                return pii_tool.fn(result, str(columns))
            except Exception as e:
                logger.warning(f"PII protection execution error: {e}")
        
        # Fallback to direct manager invocation if needed
        try:
            if self.func_manager is not None:
                return self.func_manager.pii_protection_tool(result, str(columns))
            else:
                from helper_modules.function_tools import FunctionToolsManager
                fm = FunctionToolsManager(verbose=False)
                return fm.pii_protection_tool(result, str(columns))
        except Exception:
            return result

    def _simple_routing(self, query: str) -> List[Any]:
        """Simple rule-based tool selection routing for fast and deterministic dispatch
        
        Args:
            query: Natural language question
            
        Returns:
            List of selected tool objects
        """
        q_lower = query.lower()
        selected_tools = []
        
        # 1. Database query keywords
        db_keywords = [
            'customer', 'portfolio', 'holding', 'database', 'shares', 'bought',
            'paid', 'purchase', 'profile', 'category', 'average', 'top 5',
            'risk tolerance', 'investment profile', 'sql'
        ]
        if any(kw in q_lower for kw in db_keywords):
            for t in self.function_tools:
                if 'database' in getattr(t.metadata, 'name', ''):
                    selected_tools.append(t)
        
        # 2. Market data keywords
        market_keywords = [
            'stock price', 'current price', 'stock performance', 'market data',
            'trading', 'volume', 'current stock', 'quote', 'price and'
        ]
        if any(kw in q_lower for kw in market_keywords):
            for t in self.function_tools:
                if 'market' in getattr(t.metadata, 'name', ''):
                    selected_tools.append(t)
        
        # 3. Document 10-K filing keywords
        doc_keywords = [
            '10-k', 'filing', 'annual report', 'segment', 'risk factor',
            'revenue source', 'business model', 'comprehensively'
        ]
        is_doc = any(kw in q_lower for kw in doc_keywords)
        
        if 'apple' in q_lower or 'aapl' in q_lower:
            if is_doc or ('customer' not in q_lower and 'price' not in q_lower):
                for t in self.document_tools:
                    if 'aapl' in getattr(t.metadata, 'name', '').lower():
                        selected_tools.append(t)
        
        if 'google' in q_lower or 'googl' in q_lower or 'alphabet' in q_lower:
            if is_doc or ('customer' not in q_lower and 'price' not in q_lower):
                for t in self.document_tools:
                    if 'googl' in getattr(t.metadata, 'name', '').lower():
                        selected_tools.append(t)
        
        if 'tesla' in q_lower or 'tsla' in q_lower:
            if is_doc or ('customer' not in q_lower and 'price' not in q_lower):
                for t in self.document_tools:
                    if 'tsla' in getattr(t.metadata, 'name', '').lower():
                        selected_tools.append(t)
        
        # 4. Explicit PII protection request
        if ('protect' in q_lower or 'pii' in q_lower or 'mask' in q_lower) and not selected_tools:
            for t in self.function_tools:
                if 'pii' in getattr(t.metadata, 'name', ''):
                    selected_tools.append(t)
        
        # Fallback if no specific tool matched yet
        if not selected_tools:
            for t in self.document_tools:
                if any(sym.lower() in getattr(t.metadata, 'name', '').lower() for sym in self.companies if sym.lower() in q_lower):
                    selected_tools.append(t)
            if not selected_tools and self.function_tools:
                selected_tools.append(self.function_tools[0])
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in selected_tools:
            name = getattr(t.metadata, 'name', str(t))
            if name not in seen:
                seen.add(name)
                unique.append(t)
        
        return unique

    def _intelligent_routing(self, query: str) -> List[Any]:
        """Use LLM to intelligently route query to appropriate tools
        
        Args:
            query: User's natural language query
            
        Returns:
            List of selected tool objects
        """
        all_tools = self.document_tools + self.function_tools
        if not all_tools:
            return []
        
        if self.llm is None:
            return self._simple_routing(query)
        
        tool_descriptions = []
        for i, tool in enumerate(all_tools):
            name = getattr(tool.metadata, 'name', f"tool_{i}")
            desc = getattr(tool.metadata, 'description', '')
            tool_descriptions.append(f"- {name}: {desc}")
        
        tools_str = "\n".join(tool_descriptions)
        
        prompt = f"""You are an intelligent router for a financial multi-tool assistant.
Analyze the user query and select one or more tools that should be executed to provide a complete answer.

Available Tools:
{tools_str}

Routing Guidelines:
1. Document Tools (AAPL_10k_filing_tool, GOOGL_10k_filing_tool, TSLA_10k_filing_tool):
   - Select for company SEC 10-K filings, annual reports, business segments, key risk factors, revenue breakdowns from filings.
2. Database Query Tool (database_query_tool):
   - Select for customer holdings, customer profiles, database queries, portfolio values, purchase prices.
3. Real-time Market Data Tool (finance_market_search_tool):
   - Select for real-time stock prices, latest quotes, price changes, and trading volumes.
4. Multi-Tool Coordination:
   - For queries asking to compare current market prices with customer purchase prices, select BOTH finance_market_search_tool AND database_query_tool.
   - For comprehensive queries (e.g., analyze Apple across 10-K filings, current stock, and customer ownership), select ALL relevant tools.

User Query: "{query}"

Output ONLY a JSON array containing the exact names of the selected tools.
Example: ["AAPL_10k_filing_tool", "finance_market_search_tool"]
Selected Tools:"""

        try:
            response = self.llm.complete(prompt)
            resp_text = str(response).strip()
            
            # Extract JSON list from response
            match = re.search(r'\[.*\]', resp_text, re.DOTALL)
            if match:
                selected_names = json.loads(match.group(0))
                selected_tools = [t for t in all_tools if getattr(t.metadata, 'name', '') in selected_names]
                if selected_tools:
                    return selected_tools
        except Exception as e:
            logger.warning(f"Intelligent routing LLM error: {e}, falling back to simple routing")
        
        return self._simple_routing(query)

    def _route_query(self, query: str) -> List[Tuple[str, str, Any]]:
        """Intelligently route query to appropriate tools, execute them, and collect results
        
        Args:
            query: User's natural language query
            
        Returns:
            List of tuples: (tool_name, tool_description, result)
        """
        selected_tools = self._intelligent_routing(query)
        if not selected_tools:
            selected_tools = self._simple_routing(query)
            
        results = []
        for tool in selected_tools:
            tool_name = getattr(tool.metadata, 'name', 'unknown_tool')
            tool_desc = getattr(tool.metadata, 'description', '')
            
            try:
                # 1. Document Tool (QueryEngineTool)
                if hasattr(tool, 'query_engine'):
                    tool_res = tool.query_engine.query(query)
                    result_str = str(tool_res)
                # 2. Function Tool (FunctionTool)
                elif hasattr(tool, 'fn'):
                    if tool_name == "database_query_tool":
                        raw_result = tool.fn(query)
                        # Automatic PII Protection coordination
                        result_str = self._check_and_apply_pii_protection(tool_name, raw_result)
                    elif tool_name == "pii_protection_tool":
                        result_str = tool.fn(query, "")
                    else:
                        result_str = tool.fn(query)
                elif hasattr(tool, 'call'):
                    output = tool.call(query)
                    result_str = str(getattr(output, 'content', output))
                    if tool_name == "database_query_tool":
                        result_str = self._check_and_apply_pii_protection(tool_name, result_str)
                else:
                    result_str = f"Unable to execute tool {tool_name}"
                    
                results.append((tool_name, tool_desc, result_str))
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                results.append((tool_name, tool_desc, f"Error: {e}"))
                
        return results

    def _synthesize_results(self, question: str, results: Any) -> str:
        """Synthesize results from multiple tools into a coherent, comprehensive answer
        
        Args:
            question: Original user question
            results: List of tool results (tuples, dicts, or strings)
            
        Returns:
            Comprehensive synthesized response
        """
        if not results:
            return "No data was retrieved to answer the question."
        
        # Normalize results into list of (source, content)
        items = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, tuple) and len(r) >= 3:
                    items.append((r[0], str(r[2])))
                elif isinstance(r, dict):
                    src = r.get('tool', r.get('source', 'Tool'))
                    content = r.get('result', r.get('content', str(r)))
                    items.append((src, str(content)))
                else:
                    items.append(("Tool", str(r)))
        else:
            items.append(("Tool", str(results)))
        
        if len(items) == 1:
            return items[0][1]
        
        context_parts = []
        for src, content in items:
            context_parts.append(f"### Source: {src}\n{self._fit_for_synthesis(content)}")
        results_context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert senior financial analyst.
Synthesize a comprehensive, coherent, and professional answer to the user's question by integrating data from the multiple sources provided below.

User Question: {question}

Data Sources:
{results_context}

Guidelines:
1. Provide a direct, professional, and well-structured answer.
2. Integrate figures, metrics, and insights from each source coherently.
3. Compare and contrast data where appropriate (e.g. comparing real-time stock prices with portfolio purchase prices).
4. Maintain transparency and keep any privacy notices intact.
5. For gains or losses, compute current price minus purchase price per share. A negative result is a loss; say so plainly (e.g. if purchase price was $800 and current price is $367, it is a loss of $433 per share, NOT a profit or gain).

Comprehensive Answer:"""

        try:
            if self.llm is not None:
                response = self.llm.complete(prompt)
                return str(response).strip()
        except Exception as e:
            logger.warning(f"Synthesis LLM error: {e}")
            
        return "\n\n".join([f"**{src}**:\n{content}" for src, content in items])

    def query(self, question: str, verbose: bool = None) -> str:
        """Process query with dynamic tool routing and result synthesis
        
        Args:
            question: User's financial question
            verbose: Whether to show detailed processing info
            
        Returns:
            Comprehensive answer synthesized from relevant tools
        """
        if verbose is None:
            verbose = self.verbose
        
        # Ensure tools are initialized lazily
        if not self._tools_initialized:
            self.setup()
            self._tools_initialized = True
        
        if verbose:
            print(f"🎯 Query: {question}")
        
        results = self._route_query(question)
        
        if verbose:
            print(f"   Selected {len(results)} tool(s): {[r[0] for r in results]}")
        
        if not results:
            return "I was unable to retrieve data for this query. Please check available companies and tools."
        
        if len(results) == 1:
            return str(results[0][2])
        
        synthesized = self._synthesize_results(question, results)
        return synthesized

    def list_available_tools(self) -> List[str]:
        """List all available tools in the coordinator
        
        Returns:
            List of tool descriptions
        """
        tools_list = []
        for t in self.document_tools:
            name = getattr(t.metadata, 'name', 'DocumentTool')
            desc = getattr(t.metadata, 'description', '')
            tools_list.append(f"{name}: {desc}")
        for t in self.function_tools:
            name = getattr(t.metadata, 'name', 'FunctionTool')
            desc = getattr(t.metadata, 'description', '')
            tools_list.append(f"{name}: {desc}")
        return tools_list

    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available tools with full compatibility
        
        Returns:
            Dictionary with comprehensive tool information
        """
        return {
            "document_tools": ["apple", "google", "tesla"] if len(self.document_tools) >= 3 else [],
            "function_tools": ["sql", "market", "pii"] if len(self.function_tools) >= 3 else [],
            "total_tools": len(self.document_tools) + len(self.function_tools),
            "document_tool_count": len(self.document_tools),
            "function_tool_count": len(self.function_tools)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status with full compatibility
        
        Returns:
            Dictionary with detailed status information
        """
        tool_count = len(self.document_tools) + len(self.function_tools)
        system_ready = len(self.document_tools) >= 3 and len(self.function_tools) >= 3
        
        return {
            "companies": self.companies,
            "document_tools": len(self.document_tools),
            "function_tools": len(self.function_tools),
            "total_tools": tool_count,
            "ready": system_ready,
            "architecture": "modular",
            "coordinator_ready": system_ready,
            "available_companies": ['AAPL', 'GOOGL', 'TSLA'],
            "capabilities": [
                "Document analysis (10-K filings)",
                "Database queries (customer portfolios)",
                "Real-time market data",
                "PII protection",
                "Multi-tool coordination",
                "Intelligent routing"
            ],
            "system_ready": system_ready
        }
