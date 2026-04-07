# Integration Tests

These tests require real API keys and external dependencies. They are **not run in CI**.

## Running manually

```bash
export OPENAI_API_KEY=sk-...
pip install llm-trace[langgraph]
pip install langchain-openai

python tests/integration/test_langgraph.py
python tests/integration/test_real_llm.py
```
