# BareLLM

> **Learn how LLM inference works by building a model server from scratch.**

**BareLLM** is a step-by-step guide and clean educational codebase built to demystify how Large Language Models (like ChatGPT, LLaMA, or Qwen) run under the hood.

Most production inference engines hide the inner workings of LLMs behind layers of highly optimized C++/CUDA code. **BareLLM** strips away these abstractions, letting you build a model server from the ground up using plain Python, PyTorch, and clean mathematics.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full phase-by-phase build plan.

## Getting Started

### Prerequisites

- **Python:** `>=3.13`
- **Package & Environment Manager:** [uv](https://docs.astral.sh/uv/)
- **Command Runner (Optional but recommended):** [just](https://github.com/casey/just)

### Installation

```bash
# Clone the repository
git clone https://github.com/quanhua92/barellm.git
cd barellm

# Install dependencies and set up virtual environment
uv sync

# Configure git hooks (one-time setup for code quality check on commit)
git config core.hooksPath githooks
```

## Development Workflow

We use `just` to manage development workflows. If you don't have `just` installed, you can use the corresponding `uv run` commands.

| Task | `just` Shortcut | Raw Command | Description |
|---|---|---|---|
| **Run CLI** | `just run` | `uv run barellm run` | Run generation from a prompt |
| **Serve API** | `just serve` | `uv run barellm serve` | Start the HTTP API server |
| **Pull Model** | `just pull` | `uv run barellm pull` | Download Qwen model weights |
| **List Models**| `just ls` | `uv run barellm ls` | List downloaded models |
| **Remove Model**| `just rm <model>` | `uv run barellm rm <model>` | Remove downloaded model weights |
| **Test** | `just test` | `uv run pytest` | Run the test suite |
| **Lint & Format**| `just lint` | `uv run ruff check --fix . && uv run ruff format .` | Check linting and format Python files |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
