# Contributing to AnimalLens

Thank you for your interest in contributing to **AnimalLens**!

AnimalLens is an open-source platform for animal behavior intelligence. We welcome contributions ranging from adding new species adapters and improving computer vision models, to enhancing the developer SDK, CLI, and documentation.

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/cvrvai/AnimalLens.git
cd AnimalLens
```

### 2. Set Up a Virtual Environment
```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode with Dev Dependencies
```bash
pip install --upgrade pip
pip install -e .
pip install pytest pytest-asyncio black ruff
```

### 4. Run System Diagnostics
```bash
animallens doctor
```

---

## Running Tests

Ensure all tests pass before submitting a pull request:
```bash
pytest -v
```

Code formatting checks:
```bash
ruff check .
black --check .
```

---

## Adding a New Species Adapter

AnimalLens is designed around pluggable species adapters. To add support for a new animal (e.g. `pig`, `chicken`, `fish`):

1. Create a folder in `animallens/species/<species_id>/`:
   - `config.yaml`: Species metadata, scientific name, default thresholds.
   - `taxonomy.yaml`: Hierarchical ethogram definitions.
   - `adapter.py`: Subclass of `SpeciesAdapter`.
   - `__init__.py`: Export your adapter.
2. Register your adapter in `animallens/species/registry.py` or dynamically via Python:
   ```python
   from animallens.species.registry import species_registry
   from animallens.species.pig.adapter import PigAdapter

   species_registry.register_adapter("pig", PigAdapter())
   ```
3. Add unit tests under `tests/test_species.py`.

---

## Pull Request Guidelines

1. **Keep Layer A & Layer B strictly decoupled**: Layer A (Computer Vision, Tracking, Kinematics, Temporal models) must remain 100% operational offline without requiring an LLM. Layer B (Ollama reasoning) is purely optional.
2. **Never claim unverified accuracy**: When contributing baseline or heuristic rules, mark them clearly as development implementations.
3. **Preserve Pydantic Schema compatibility**: All behavior events must conform to the standard `BehaviorEvent` schema.
4. **Include tests**: Add unit tests for any new features, bug fixes, or species adapters.

---

## License

By contributing to AnimalLens, you agree that your contributions will be licensed under the **Apache License 2.0**.
