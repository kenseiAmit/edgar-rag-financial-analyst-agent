# FinTool Analyst: Multi-Tool SEC 10-K RAG, NL-to-SQL & PII-Guarded Equity Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/RAG-LlamaIndex%20VectorStoreIndex-7C3AED?style=flat-square)
![Tests](https://img.shields.io/badge/Verification-79%2F79%20Checks%20Passing-16A34A?style=flat-square)
![PII Guard](https://img.shields.io/badge/SQL%20Security-Read--Only%20%2B%20Auto%20PII%20Masking-0D9488?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-slate?style=flat-square)

Autonomous equity and portfolio research coordinator built with **LlamaIndex** and **OpenAI**—unifying **FY2024 SEC Form 10-K vector retrieval** (`AAPL`, `GOOGL`, `TSLA`), **schema-grounded read-only SQLite NL-to-SQL generation** (`MAX_ROWS = 25`), **live Yahoo Finance market telemetry**, and **pre-LLM PII redaction**.

![FinTool Verification & Guardrails](docs/images/fintool_verification_and_guardrails.png)

---

## 1. Multi-Source Tool Routing & PII Interception Architecture

```mermaid
flowchart TD
    Q["Analyst Multi-Source Query"] --> C["FinancialAgentCoordinator<br/>(ReAct / Function-Calling Router)"]
    C --> T1["AAPL / GOOGL / TSLA 10-K Tools<br/>(VectorStoreIndex · chunk=1024/128)"]
    C --> T2["query_financial_database<br/>(PRAGMA Schema -> SELECT-Only SQL)"]
    T2 --> M["Deterministic PII Masker<br/>(j***@domain.com · ***-***-1234 · J*** D***)"]
    M --> C
    C --> T3["get_company_market_data / compare_stocks<br/>(Live Yahoo Finance + Offline Fallback)"]
    C --> R["Synthesized Grounded Investment Brief"]
```

---

## 2. Sample PII-Masked SQL & Cross-Source Benchmark Trace

```text
$ python financial_agent.py --status
[OK] SEC 10-K Indices Ready: AAPL_10K_2024.pdf, GOOGL_10K_2024.pdf, TSLA_10K_2024.pdf
[OK] SQLite Database Connected: data/financial.db (clients, portfolios, holdings, transactions)
[OK] Active Tool Registry: 6/6 tools registered

# SQL PII Interception Example (Executed Before LLM Observation):
Raw DB Row:   {"client_name": "Jane Doe", "email": "jane.doe@example.com", "phone": "555-234-5678", "aum": 425000}
Masked Tool:  {"client_name": "J*** D***", "email": "j***@example.com",     "phone": "***-***-5678", "aum": 425000}
```

Full 26-cell interactive walkthrough with live SEC 10-K + SQL + market synthesis outputs: [`financial_agent_walkthrough.ipynb`](financial_agent_walkthrough.ipynb).

---

## 3. Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python financial_agent.py --status
python financial_agent.py --benchmark
pytest tests/ -v
```

## License

MIT License — see [LICENSE](LICENSE).
