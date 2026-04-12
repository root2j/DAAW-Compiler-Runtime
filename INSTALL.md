# Installation Guide (Linux / macOS / WSL)

## Prerequisites

- Python 3.9+ (`python3 --version`)
- pip (`pip3 --version`)
- git (`git --version`)
- Optional: Ollama for local LLM inference

---

## 1. Clone the Repository

```bash
git clone https://github.com/root2j/DAAW-Compiler-Runtime.git
cd DAAW-Compiler-Runtime
```

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
# Core + UI + dev tools
pip install -e ".[ui,dev]"

# Or from requirements.txt (pinned versions)
pip install -r requirements.txt
```

## 4. Set the Python Path

```bash
export PYTHONPATH=src

# Add to your shell profile to make it permanent:
echo 'export PYTHONPATH=$HOME/DAAW-Compiler-Runtime/src' >> ~/.bashrc
source ~/.bashrc
```

## 5. Configure Environment

```bash
cp .env.example .env
nano .env   # or vim .env
```

### Cloud Providers (set whichever you have)

```env
GROQ_API_KEY=gsk_...          # Free tier at console.groq.com
GEMINI_API_KEY=AIza...         # Free tier at aistudio.google.dev
OPENAI_API_KEY=sk-...          # console.platform.openai.com
ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com
```

### Local LLM via Ollama (no API key needed)

```env
GATEWAY_URL=http://localhost:11434/v1
GATEWAY_MODEL=gemma4:e4b
```

### Optional: Webhook Notifications

```env
NOTIFY_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook
NOTIFY_WEBHOOK_TYPE=discord    # or: slack, generic
```

## 6. Install Ollama (Optional, for Local LLMs)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull a model (pick one)
ollama pull gemma4:e4b           # 9GB, recommended for planning + execution
ollama pull gemma4:e2b-it-q4_K_M # 7GB, faster, simpler tasks only
ollama pull llama3.2              # 2GB, very fast, limited planning

# Verify it works
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

### Ollama as a systemd service (auto-start on boot)

```bash
# Ollama installs its own service, just enable it:
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

### GPU Support (NVIDIA)

```bash
# Verify CUDA is available
nvidia-smi

# Ollama auto-detects NVIDIA GPUs. No extra config needed.
# For AMD GPUs, install ROCm: https://ollama.ai/blog/amd-preview
```

---

## 7. Verify Installation

```bash
# Run unit tests (no API keys needed)
python -m pytest tests/ -x -q -m "not integration"

# Test imports
python -c "from daaw.schemas.workflow import WorkflowSpec; print('OK')"

# Launch demo UI (no API keys needed)
python -m daaw ui
# Opens at http://localhost:8501
```

## 8. Run the Pipeline

```bash
# With local LLM (Ollama)
python -m daaw run --goal "Research AI trends and write a summary" --provider gateway

# With Groq (free cloud API)
python -m daaw run --goal "Plan a trip to Japan with costs" --provider groq

# Interactive mode
python -m daaw run --provider gateway

# Launch UIs
python -m daaw ui     # Dashboard (port 8501)
python -m daaw demo   # Demo UI (port 8502)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'daaw'`

```bash
export PYTHONPATH=src
# Or install in editable mode:
pip install -e .
```

### `Provider 'gateway' not available`

```bash
# Check Ollama is running
curl http://localhost:11434/v1/models

# If not, start it:
ollama serve &

# Check .env has GATEWAY_URL set
grep GATEWAY .env
```

### `Connection refused` to Ollama

```bash
# Ollama defaults to localhost only. For remote access:
OLLAMA_HOST=0.0.0.0 ollama serve

# Or set in systemd:
sudo systemctl edit ollama
# Add: Environment="OLLAMA_HOST=0.0.0.0"
```

### Streamlit won't start

```bash
pip install streamlit plotly
# Or:
pip install -e ".[ui]"
```

### GPU out of memory

```bash
# Check VRAM usage
nvidia-smi

# Use a smaller model
ollama pull gemma4:e2b-it-q4_K_M  # 7GB instead of 9GB

# Or use CPU-only (slower but works)
OLLAMA_NUM_GPU=0 ollama serve
```

### Tests fail with `test_task_timeout`

This test is timing-dependent and flaky on fast machines. Skip it:

```bash
python -m pytest tests/ -x -q -m "not integration" -k "not test_task_timeout"
```

---

## Project Structure

```
DAAW-Compiler-Runtime/
├── src/daaw/           # Main package
│   ├── cli/            # CLI entry points (run, legacy, ui, demo)
│   ├── compiler/       # Goal --> WorkflowSpec + JSON repair
│   ├── engine/         # DAG executor, circuit breaker, context pruner
│   ├── agents/         # Agent base, factory, registry, 6 built-in agents
│   ├── llm/            # Unified LLM client + 5 providers
│   ├── critic/         # Task evaluation + runtime patching
│   ├── schemas/        # Pydantic data models
│   ├── store/          # Artifact store (async key-value)
│   ├── tools/          # Tool registry, real tools, webhook tools
│   ├── integrations/   # Webhook notifications
│   └── ui/             # Dashboard (app.py) + Demo UI (demo_app.py)
├── tests/              # 145+ tests
├── .env                # Your API keys (not committed)
├── requirements.txt    # Pinned dependencies
├── pyproject.toml      # Package config
└── README.md           # Overview
```

---

## Uninstall

```bash
deactivate              # Exit virtual environment
rm -rf venv             # Remove venv
cd .. && rm -rf DAAW-Compiler-Runtime  # Remove project

# Remove Ollama (if installed)
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
rm -rf ~/.ollama
```
