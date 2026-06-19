# AwareKernel

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/sachn-cs/aware-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/aware-kernel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-green.svg)](https://mypy-lang.org/)

Refresh-aware hybrid continuous-discrete low-rank kernel learning for scalable, adaptive kernel regression.

## Features

- **Explicit-feature kernel system** with PSD guarantee (`K = Phi Phi^T >= 0`)
- **Global Nyström basis** with soft-truncated whitening for stable low-rank approximation
- **Local corrective features** with residual-aware anchor sampling and k-NN sparsity
- **Residual orthogonalization** ensuring local features live in the global nullspace
- **Feature calibration and fusion** via logistic gate balancing global/local contributions
- **Refresh controller** with drift-aware trigger, cooldown, warmup, hysteresis, and amortized budget
- **Two memory modes**: cached O(nm) and streamed O(m^2) for different scale regimes
- **Numerical stabilization** including eigenvalue clipping, soft spectral truncation, and Cholesky with jitter fallback
- **Sklearn-compatible API** (`fit`, `predict`, `score`)

## Installation

```bash
# Clone the repository
git clone https://github.com/sachn-cs/aware-kernel.git
cd aware-kernel

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, scikit-learn >= 1.3

## Usage

```python
import numpy as np
from aware_kernel import AwareKernelEstimator

# Generate synthetic data
rng = np.random.default_rng(42)
X_train = rng.standard_normal((200, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

X_test = rng.standard_normal((50, 4))
y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(50)

# Fit the model
model = AwareKernelEstimator(
    embedding_dim=4,
    m_g=32,
    m_l=8,
    lambda_reg=1e-2,
    max_steps=50,
    seed=42,
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R^2 score: {model.score(X_test, y_test):.4f}")
```

### Memory Modes

```python
# Cached (default) - O(nm) memory, simpler
model = AwareKernelEstimator(memory_mode="cached")

# Streamed - O(m^2) memory, scales to larger datasets
model = AwareKernelEstimator(memory_mode="streamed")
```

### Ablation Studies

```python
# Disable specific components for ablation
model = AwareKernelEstimator(
    disable_refresh=True,            # No discrete refreshes
    disable_orthogonalization=True,  # Skip orthogonalization
    disable_diversity_penalty=True,  # Remove diversity regularization
)
```

See [docs/getting-started.md](docs/getting-started.md) for detailed configuration options.

## Project Structure

```
aware-kernel/
├── aware_kernel/           # Main package
│   ├── aware/              # Core types, config, state, exceptions
│   ├── embedding/          # Dense embedder and projection matrix
│   ├── global_basis/       # Nyström landmark selection and whitening
│   ├── local_corrective/   # Anchor sampling, sparse features, orthogonalization
│   ├── fusion/             # Calibration, gating, fused feature building
│   ├── solver/             # Ridge regression (Cholesky / PCG)
│   ├── memory/             # Cached and streamed accumulators
│   ├── refresh/            # Drift, budget, controller, refresh pipeline
│   ├── training/           # Training loop, objectives, callbacks
│   ├── inference/          # Mean and variance prediction
│   ├── evaluation/         # Datasets, baselines, metrics, experiment runner
│   ├── utils/              # Linear algebra, numerics, sampling utilities
│   └── api.py              # Sklearn-compatible public API
├── tests/                  # Test suite (unit, numerical, integration)
├── examples/               # Real-world evaluation examples
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

## Development

| Command | Description |
|---------|-------------|
| `pip install -e ".[dev]"` | Install with dev dependencies |
| `pytest tests/ -v -p no:asyncio` | Run full test suite |
| `pytest tests/unit/ -v` | Run unit tests only |
| `ruff check .` | Lint with ruff |
| `black .` | Format with black |
| `mypy --strict aware_kernel/` | Type check with mypy |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Numerical | NumPy, SciPy |
| Machine Learning | scikit-learn |
| Linting | ruff, black |
| Type Checking | mypy (strict) |
| Testing | pytest, pytest-cov, hypothesis |
| Pre-commit | pre-commit hooks |

## Architecture

The method separates model parameters into two groups:

- **Continuous parameters** (`theta`, `R`): Updated every training step via gradient descent
- **Discrete parameters** (`Z`, `A`, `M_g`, `c_g`, `c_l`, `d`): Refreshed only when drift exceeds a threshold

This hybrid approach makes the method efficient for streaming/large-batch settings, because the expensive discrete refresh is triggered adaptively rather than every step.

### Mathematical Guarantees

1. **PSD kernel**: `K = Phi Phi^T` is positive semidefinite
2. **Rank bound**: `rank(K) <= r_g + m_l`
3. **SPD normal equations**: `S = Phi^T Phi + lambda I` is symmetric positive definite
4. **Orthogonalization**: `Phi_g^T Phi_l_perp ~= 0` up to ridge regularization
5. **Calibration stability**: calibration scalars bounded away from zero

See [docs/architecture.md](docs/architecture.md) for full design rationale and extension points.

## Roadmap

- [ ] GPU solver backend (CuPy integration)
- [ ] Learned embedding functions (neural embedder support)
- [ ] Streaming/incremental fit API
- [ ] Classification support
- [ ] Benchmark suite with published results
- [ ] PyPI distribution
- [ ] Documentation website (MkDocs)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Branch naming conventions
- Commit message format (Conventional Commits)
- Pull request process
- Coding standards

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to its terms.

## Security

For reporting security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
