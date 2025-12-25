# Prompt Injection Defense Experiments

This directory contains the experimental framework for evaluating prompt injection defenses in full-stack LLM applications.

## Project Structure

```
experiments/
├── src/
│   ├── models/          # Pydantic data models
│   ├── layers/          # Defense layer implementations
│   ├── config.py        # Configuration management
│   └── database.py      # SQLite persistence
├── data/
│   ├── attack_prompts.py  # Attack dataset (50+ prompts)
│   └── experiments.db     # SQLite database (created automatically)
├── frontend/            # Simple web UI
├── notebooks/           # Jupyter notebooks for analysis
├── results/             # Experiment results and figures
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd experiments
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and Configure Ollama

Follow the official Ollama installation guide: https://ollama.ai/download

Then download the required models:

```bash
ollama pull llama3
ollama pull mistral
```

Test that Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

### 3. Verify Installation

```bash
python -c "import fastapi, pydantic, sentence_transformers; print('Dependencies OK')"
```

## Research Questions

**RQ1**: How do prompt injection attacks propagate across different layers of a full-stack web application?

**RQ2**: Which system-level trust boundary violations enable successful prompt injection?

**RQ3**: How can coordinated workflow-level defenses reduce attack success compared to isolated mitigations?

## Experiments

### Experiment 1: Layer Propagation Analysis (RQ1)

Tests 4 configurations:
- Version A: No defenses (baseline)
- Version B: Layer 2 only (semantic filtering)
- Version C: Layers 2-3 (+ context isolation)
- Version D: Layers 2-5 (full workflow)

**Run**: `python run_experiment1.py`

### Experiment 2: Trust Boundary Violation Analysis (RQ2)

Tests architectural trust boundary conditions:
- Shared prompt string (vulnerable)
- Separate system/user variables
- Metadata-tagged prompts
- Hard isolation (secure)

**Run**: `python run_experiment2.py`

### Experiment 3: Coordinated vs Isolated Defenses (RQ3)

Tests 5 defense configurations:
- D1: Keyword filtering only
- D2: Semantic filtering only
- D3: Guardrail LLM only
- D4: Combined filtering + guardrail
- D5: Full workflow

**Run**: `python run_experiment3.py`

### Experiment 4: Layer Ablation Study

Measures individual layer contribution by removing one at a time.

**Run**: `python run_experiment4.py`

## Attack Dataset

The dataset includes 50+ attack prompts across categories:
- Direct injection (10 prompts)
- Semantic/polite injection (10 prompts)
- Context override (10 prompts)
- Encoding-based attacks (3 prompts)
- Multi-turn attacks (4 prompts)
- Jailbreak attempts (5 prompts)

Plus 10 benign prompts for false positive testing.

## Defense Layers

1. **Layer 1 - Request Boundary**: Input validation (length, encoding)
2. **Layer 2 - Semantic Analysis**: Embedding-based attack detection
3. **Layer 3 - Context Isolation**: System/user prompt separation
4. **Layer 4 - LLM Interaction**: Guardrail model validation
5. **Layer 5 - Output Validation**: Policy violation detection

## Data Collection

All experiment results are stored in `experiments.db` (SQLite) with:
- Request metadata
- Layer-by-layer decisions
- Final outcomes
- Timing information
- Ground truth labels

## Analysis

Use Jupyter notebooks in `notebooks/` to:
- Generate result tables
- Create visualizations (bar charts, Sankey diagrams)
- Compute statistical metrics
- Answer research questions

## Reproducibility

- Fixed random seed: 42
- Model versions logged
- All prompts version controlled
- Configuration stored in database

## Next Steps

1. ✅ Setup environment
2. ⬜ Implement defense layers
3. ⬜ Create backend API
4. ⬜ Run experiments 1-4
5. ⬜ Analyze results
6. ⬜ Write paper results section

## License

Academic use only - for the research paper "Evaluating and Mitigating Prompt Injection in Full-Stack Web Applications"
