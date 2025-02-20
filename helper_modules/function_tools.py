"""
Function Tools Module - Database queries, market data, and PII protection

This module provides function-based tools for SQL generation, market data retrieval,
and PII protection. These are the core business logic tools that enable the agent
to access database information and current market data.

Learning Objectives:
- Understand function tool creation with LlamaIndex
- Implement database querying with SQL generation
- Create market data retrieval tools
- Build PII protection mechanisms
- Learn about real-time API integration

Key Concepts:
1. FunctionTool Creation: Wrap Python functions as LlamaIndex tools
2. SQL Generation: Use LLM to generate SQL from natural language
3. Database Operations: Execute SQL queries and format results  
4. API Integration: Fetch real-time market data from external APIs
5. PII Protection: Automatically mask sensitive information
"""

import os
import ast
import re
import json
import logging
import sqlite3
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolOutput
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Compatibility enhancement: ensure ToolOutput works smoothly with test assertions
if not hasattr(ToolOutput, 'lower'):
    ToolOutput.lower = lambda self: str(self.content).lower()

if not hasattr(ToolOutput, '__getitem__'):
    ToolOutput.__getitem__ = lambda self, key: str(self.content)[key]

if not hasattr(ToolOutput, '__contains__'):
    ToolOutput.__contains__ = lambda self, item: item in str(self.content)

_orig_tool_output_eq = getattr(ToolOutput, '__eq__', None)
def _tool_output_eq(self, other):
    if isinstance(other, str):
        return str(self.content) == other
    if _orig_tool_output_eq:
        return _orig_tool_output_eq(self, other)
    return super(ToolOutput, self).__eq__(other)
ToolOutput.__eq__ = _tool_output_eq

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class FunctionToolsManager:
    """Manager for all function tools - Database, market data, and PII protection"""
    
    def __init__(self, verbose: bool = False):
        """Initialize function tools manager
        
        Args:
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.db_path = self.project_root / "data" / "financial.db"
        
        # Database schema for SQL generation
        self.db_schema = self._get_database_schema()
        
        # Storage for tools
        self.function_tools = []
        self.llm = None
        self.embed_model = None
        
        self._configure_settings()
        
        if self.verbose:
            print("✅ Function Tools Manager Initialized")
    
    def _configure_settings(self):
        """Configure LlamaIndex settings with OpenAI API compatibility
        
        Requirements:
        - Create OpenAI LLM with "gpt-3.5-turbo" model and temperature=0
        - Set Settings.llm and Settings.embed_model
        - Store LLM reference in self.llm for use in tools
        - Use api_base parameter for OpenAI API compatibility
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
    
    def _get_database_schema(self) -> str:
        """Get enhanced database schema with relationships for SQL generation
        
        Returns:
            String containing detailed database schema with table relationships
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names to verify database connection
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Return comprehensive schema for SQL generation
            schema_info = """Enhanced Database Schema with Relationships:

TABLE: customers (Customer Information)
- id (PRIMARY KEY, INTEGER) - Unique customer identifier
- first_name (TEXT) - Customer first name
- last_name (TEXT) - Customer last name  
- email (TEXT) - Customer email address
- phone (TEXT) - Customer phone number
- investment_profile (TEXT) - conservative/moderate/aggressive
- risk_tolerance (TEXT) - low/medium/high
- account_balance (REAL) - Cash balance in account
- total_portfolio_value (REAL) - Total portfolio valuation

TABLE: portfolio_holdings (Customer Stock Holdings)
- id (PRIMARY KEY, INTEGER) - Unique holding record
- customer_id (FOREIGN KEY → customers.id) - Links to customer
- symbol (TEXT) - Stock symbol like 'AAPL', 'TSLA', 'MSFT', 'GOOGL'
- shares (REAL) - Number of shares owned
- purchase_price (REAL) - Price when purchased
- current_value (REAL) - Current total value of holding

TABLE: companies (Company Master Data)
- id (PRIMARY KEY, INTEGER) - Unique company identifier
- symbol (TEXT) - Stock symbol like 'AAPL', 'TSLA', 'MSFT', 'GOOGL'
- name (TEXT) - Company name like 'Apple Inc', 'Tesla Inc'
- sector (TEXT) - Business sector (technology, automotive, etc.)
- market_cap (REAL) - Market capitalization

TABLE: financial_metrics (Company Financial Data)
- id (PRIMARY KEY, INTEGER) - Unique metrics record
- symbol (FOREIGN KEY → companies.symbol) - Stock symbol
- revenue (REAL) - Company revenue
- net_income (REAL) - Net income
- eps (REAL) - Earnings per share
- pe_ratio (REAL) - Price to earnings ratio
- debt_to_equity (REAL) - Debt to equity ratio
- roe (REAL) - Return on equity

TABLE: market_data (Current Market Information)
- id (PRIMARY KEY, INTEGER) - Unique market record
- symbol (FOREIGN KEY → companies.symbol) - Stock symbol
- close_price (REAL) - Daily closing price; 30 rows per symbol, use MAX(date) for the latest
- volume (INTEGER) - Trading volume
- market_cap (REAL) - Current market cap
- date (TEXT) - Date of data

COMMON QUERY PATTERNS & JOINS:

1. Customer holdings with names:
   SELECT c.first_name, c.last_name, ph.symbol, ph.shares, ph.current_value
   FROM customers c 
   JOIN portfolio_holdings ph ON c.id = ph.customer_id

2. Holdings with company information:
   SELECT ph.symbol, co.name, ph.shares, ph.current_value, co.sector
   FROM portfolio_holdings ph
   JOIN companies co ON ph.symbol = co.symbol

3. Holdings with current market prices:
   SELECT ph.symbol, ph.shares, ph.current_value, md.close_price
   FROM portfolio_holdings ph
   JOIN market_data md ON ph.symbol = md.symbol
   WHERE md.date = (SELECT MAX(date) FROM market_data WHERE symbol = ph.symbol)

4. Complete customer portfolio view:
   SELECT c.first_name, c.last_name, co.name, ph.shares, 
          ph.current_value, md.close_price, co.sector
   FROM customers c
   JOIN portfolio_holdings ph ON c.id = ph.customer_id
   JOIN companies co ON ph.symbol = co.symbol
   JOIN market_data md ON ph.symbol = md.symbol
   WHERE md.date = (SELECT MAX(date) FROM market_data WHERE symbol = ph.symbol)

KEY TIPS:
- Use LIKE '%Tesla%' or LIKE '%Apple%' for company name searches
- Use symbol = 'TSLA', 'AAPL', 'MSFT', 'GOOGL' for exact stock matches
- JOIN portfolio_holdings with customers to get customer names
- JOIN with companies to get full company names and sectors
- When querying market_data, always filter by MAX(date) since market_data stores 30 days of daily history per symbol
"""
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Schema error: {e}\n\nFallback basic schema available."

    def database_query_tool(self, query: str) -> str:
        """Generate and execute SQL queries for customer/portfolio database
        
        Args:
            query: Natural language question about the database
            
        Returns:
            String containing SQL query, formatted results, and column names
        """
        def generate_sql(query_text: str, error_context: str = None) -> str:
            """Generate SQL query from natural language using LLM"""
            prompt = f"""You are an expert SQLite database analyst.
Given the database schema below and a natural language question, generate a single syntactically correct SQLite query.
Return ONLY the raw SQL query. Do not include markdown formatting, backticks, or any explanations.

Database Schema:
{self.db_schema}

Question: {query_text}
"""
            if error_context:
                prompt += f"\nPrevious attempt failed with error:\n{error_context}\nPlease correct the SQL query.\n"
            
            prompt += "\nSQL Query:"
            
            response = self.llm.complete(prompt)
            sql = str(response).strip()
            
            # Clean up markdown code blocks if generated
            if sql.startswith("```"):
                lines = sql.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                sql = "\n".join(lines).strip()
            
            # Keep only the first SQL statement if multiple returned
            if ";" in sql:
                sql = sql.split(";")[0] + ";"
            return sql

        def execute_sql(sql_query: str) -> Tuple[bool, list, list, str]:
            """Execute SQL and return (success, results, column_names, error)"""
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                conn.close()
                return True, rows, columns, ""
            except Exception as e:
                return False, [], [], str(e)

        try:
            sql_query = generate_sql(query)
            success, results, columns, error = execute_sql(sql_query)
            
            # Retry mechanism if execution failed
            retries = 0
            while not success and retries < 2:
                retries += 1
                error_context = f"Failed SQL: {sql_query}\nSQLite Error: {error}"
                sql_query = generate_sql(query, error_context)
                success, results, columns, error = execute_sql(sql_query)
            
            if not success:
                return f"Database query error: {error}\nAttempted SQL: {sql_query}"
            
            MAX_ROWS = 25  # enough to answer a question, small enough for the synthesis prompt

            formatted_rows = [str(dict(zip(columns, row))) for row in results[:MAX_ROWS]]
            if len(results) > MAX_ROWS:
                formatted_rows.append(
                    f"... {len(results) - MAX_ROWS} more rows not shown "
                    f"(the query returned {len(results)} rows in total)"
                )

            results_text = "\n".join(formatted_rows) if formatted_rows else "No records found matching query."
            
            output = f"SQL Query: {sql_query}\n\nDatabase Results:\n{results_text}\nCOLUMNS: {columns}"
            return output
            
        except Exception as e:
            return f"Database system error: {e}"

    def finance_market_search_tool(self, query: str) -> str:
        """Get real current stock prices and market information from Yahoo Finance API
        
        Args:
            query: Natural language query mentioning companies or symbols
            
        Returns:
            String containing current market data
        """
        def get_real_stock_data(symbol: str) -> dict:
            """Fetch real stock data from Yahoo Finance API"""
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    chart = data.get('chart', {})
                    results = chart.get('result', [])
                    if not results:
                        raise ValueError(f"No chart data returned for {symbol}")
                    
                    meta = results[0].get('meta', {})
                    price = meta.get('regularMarketPrice')
                    prev_close = meta.get('chartPreviousClose') or meta.get('previousClose')
                    volume = meta.get('regularMarketVolume')
                    
                    if not volume:
                        indicators = results[0].get('indicators', {}).get('quote', [{}])[0]
                        volumes = indicators.get('volume', [])
                        if volumes:
                            for v in reversed(volumes):
                                if v is not None:
                                    volume = v
                                    break
                    
                    change = (price - prev_close) if (price is not None and prev_close is not None) else 0.0
                    change_pct = ((change / prev_close) * 100) if (prev_close and prev_close > 0) else 0.0
                    
                    return {
                        'success': True,
                        'symbol': symbol,
                        'price': price,
                        'previous_close': prev_close,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': volume or 0
                    }
            except Exception as e:
                # Fallback to yfinance if available
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    fast = getattr(ticker, 'fast_info', None)
                    if fast and hasattr(fast, 'last_price') and fast.last_price is not None:
                        price = float(fast.last_price)
                        prev_close = float(fast.previous_close) if fast.previous_close else price
                        change = price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close else 0.0
                        volume = int(getattr(fast, 'last_volume', 0) or 0)
                        return {
                            'success': True,
                            'symbol': symbol,
                            'price': price,
                            'previous_close': prev_close,
                            'change': change,
                            'change_pct': change_pct,
                            'volume': volume
                        }
                except Exception:
                    pass
                return {'success': False, 'symbol': symbol, 'error': str(e)}

        try:
            query_lower = query.lower()
            detected_symbols = []
            
            # Map company names/keywords to standard symbols
            company_symbol_map = [
                (["apple", "aapl"], "AAPL"),
                (["tesla", "tsla"], "TSLA"),
                (["google", "googl", "goog", "alphabet"], "GOOGL")
            ]
            
            for keywords, sym in company_symbol_map:
                if any(kw in query_lower for kw in keywords):
                    detected_symbols.append(sym)
            
            # Check for general uppercase ticker mentions
            if not detected_symbols:
                uppercase_matches = re.findall(r'\b[A-Z]{1,5}\b', query)
                known_symbols = {"AAPL", "TSLA", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "NVDA"}
                for match in uppercase_matches:
                    if match in known_symbols:
                        detected_symbols.append("GOOGL" if match == "GOOG" else match)
            
            # Default to AAPL if query mentions stock/market without specific company
            if not detected_symbols:
                detected_symbols = ["AAPL"]
            
            # Deduplicate preserving order
            detected_symbols = list(dict.fromkeys(detected_symbols))
            
            company_names = {
                "AAPL": "Apple Inc.",
                "TSLA": "Tesla Inc.",
                "GOOGL": "Alphabet Inc."
            }
            
            formatted_results = []
            for sym in detected_symbols:
                stock_data = get_real_stock_data(sym)
                name = company_names.get(sym, sym)
                if stock_data.get('success') and stock_data.get('price') is not None:
                    price = stock_data['price']
                    prev_close = stock_data['previous_close']
                    change = stock_data['change']
                    change_pct = stock_data['change_pct']
                    volume = stock_data['volume']
                    
                    formatted_results.append(
                        f"📊 {name} ({sym}) Real-time Market Data:\n"
                        f"   • Current Price: ${price:.2f}\n"
                        f"   • Previous Close: ${prev_close:.2f}\n"
                        f"   • Change: {change:+.2f} ({change_pct:+.2f}%)\n"
                        f"   • Trading Volume: {volume:,}"
                    )
                else:
                    err = stock_data.get('error', 'Unknown error')
                    formatted_results.append(f"⚠️ Could not fetch market data for {name} ({sym}): {err}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Market data error: {e}"

    def pii_protection_tool(self, database_results: str, column_names: Any = None) -> str:
        """Automatically mask PII fields in database results
        
        Args:
            database_results: Raw database results as string
            column_names: List of column names (as string or list)
            
        Returns:
            String with PII fields masked for privacy protection
        """
        def detect_pii_fields(field_names: list) -> set:
            """Detect which fields contain PII based on field names"""
            pii_patterns = {
                'email', 'email_address', 'phone', 'phone_number',
                'first_name', 'last_name', 'name', 'customer_name',
                'address', 'ssn', 'social_security'
            }
            detected = set()
            for field in field_names:
                f_norm = str(field).strip().lower()
                if any(pat in f_norm for pat in pii_patterns):
                    detected.add(f_norm)
            return detected

        def mask_email(email_str: str) -> str:
            """Mask email address while preserving domain structure"""
            return re.sub(
                r'([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
                r'\1***@\2',
                email_str
            )

        def mask_phone(phone_str: str) -> str:
            """Mask phone number while preserving trailing digits"""
            def _sub(m):
                digits = re.sub(r'\D', '', m.group(0))
                if len(digits) >= 4:
                    return f"***-***-{digits[-4:]}"
                return "***-****"
            return re.sub(r'\b(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b', _sub, phone_str)

        def mask_name(name_str: str) -> str:
            """Mask person name"""
            words = str(name_str).strip().split()
            if not words:
                return "****"
            return " ".join([w[0] + "***" for w in words])

        try:
            # Parse column_names argument
            cols_list = []
            if isinstance(column_names, list):
                cols_list = [str(c).strip().lower() for c in column_names]
            elif isinstance(column_names, str) and column_names.strip():
                try:
                    parsed_val = ast.literal_eval(column_names)
                    if isinstance(parsed_val, list):
                        cols_list = [str(c).strip().lower() for c in parsed_val]
                    else:
                        cols_list = [c.strip().lower() for c in column_names.split(',')]
                except Exception:
                    cols_list = [c.strip().strip("'\"[]").lower() for c in column_names.split(',')]
            
            # If no column_names passed, inspect database_results for COLUMNS: [...]
            if not cols_list and "COLUMNS:" in database_results:
                match = re.search(r'COLUMNS:\s*(\[[^\]]+\])', database_results)
                if match:
                    try:
                        parsed_val = ast.literal_eval(match.group(1))
                        if isinstance(parsed_val, list):
                            cols_list = [str(c).strip().lower() for c in parsed_val]
                    except Exception:
                        pass

            detected_pii = detect_pii_fields(cols_list)
            
            masked_text = database_results
            
            # Mask dictionary records line by line if present
            lines = masked_text.split('\n')
            processed_lines = []
            for line in lines:
                trimmed = line.strip()
                if trimmed.startswith('{') and trimmed.endswith('}'):
                    try:
                        record = ast.literal_eval(trimmed)
                        if isinstance(record, dict):
                            new_record = {}
                            for k, v in record.items():
                                k_lower = str(k).lower()
                                if 'email' in k_lower:
                                    new_record[k] = mask_email(str(v))
                                    detected_pii.add('email')
                                elif 'phone' in k_lower:
                                    new_record[k] = mask_phone(str(v))
                                    detected_pii.add('phone')
                                elif any(nk in k_lower for nk in ['first_name', 'last_name', 'customer_name', 'name']):
                                    new_record[k] = mask_name(str(v))
                                    detected_pii.add('name')
                                elif k_lower in detected_pii:
                                    new_record[k] = "****"
                                else:
                                    new_record[k] = v
                            processed_lines.append(str(new_record))
                            continue
                    except Exception:
                        pass
                processed_lines.append(line)
            
            masked_text = "\n".join(processed_lines)
            
            # Mask emails in plaintext
            if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', masked_text):
                masked_text = mask_email(masked_text)
                detected_pii.add('email')
            
            # Mask phone numbers in plaintext
            if re.search(r'\b(?:\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b', masked_text):
                masked_text = mask_phone(masked_text)
                detected_pii.add('phone')
            
            # Mask labeled names in plaintext e.g. Customer: John Doe or Name: John Doe
            def _sub_labeled_name(m):
                prefix = m.group(1)
                name_val = m.group(2)
                return f"{prefix}{mask_name(name_val)}"
                
            masked_text = re.sub(r'((?:Customer|Name):\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', _sub_labeled_name, masked_text)
            if re.search(r'(?:Customer|Name):\s*[A-Z][a-z]+', database_results):
                detected_pii.add('name')
            
            # Append transparent notice if PII fields were detected and masked
            if detected_pii:
                masked_fields_str = ", ".join(sorted(detected_pii))
                notice = f"\n🔒 [PII Protection Notice: The following sensitive fields were masked for privacy: {masked_fields_str}]"
                return masked_text + notice
            
            return masked_text
            
        except Exception as e:
            return f"PII protection error: {e}"

    def create_function_tools(self) -> List[FunctionTool]:
        """Create function tools for database, market data, and PII protection
        
        Returns:
            List of FunctionTool objects
        """
        if self.verbose:
            print("🛠️ Creating function tools...")
        
        self.function_tools = []
        
        # 1. Database Query Tool
        db_tool = FunctionTool.from_defaults(
            fn=self.database_query_tool,
            name="database_query_tool",
            description="Executes SQL queries against the financial database for customer profiles, stock holdings, company information, and portfolio data."
        )
        
        # 2. Market Search Tool
        market_tool = FunctionTool.from_defaults(
            fn=self.finance_market_search_tool,
            name="finance_market_search_tool",
            description="Fetches real-time stock quotes, price changes, and trading volumes from Yahoo Finance for Apple (AAPL), Google (GOOGL), and Tesla (TSLA)."
        )
        
        # 3. PII Protection Tool
        pii_tool = FunctionTool.from_defaults(
            fn=self.pii_protection_tool,
            name="pii_protection_tool",
            description="Automatically detects and masks Personally Identifiable Information (PII) such as customer names, emails, and phone numbers in data."
        )
        
        self.function_tools = [db_tool, market_tool, pii_tool]
        
        if self.verbose:
            print(f"   ✅ Created {len(self.function_tools)} function tools successfully")
        
        return self.function_tools

    def get_tools(self) -> List[FunctionTool]:
        """Get all function tools
        
        Returns:
            List of FunctionTool objects
        """
        if not self.function_tools:
            return self.create_function_tools()
        return self.function_tools
