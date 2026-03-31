# Contributing to BioPAT

We welcome contributions to the BioPAT project! As an open science effort, your input helps advance the state of biomedical IR.

## How to Contribute

1. **Bug Reports**: Open an issue on GitHub describing the bug and steps to reproduce.
2. **Feature Requests**: Discuss new features in the GitHub Issues.
3. **Pull Requests**:
    - Fork the repository.
    - Create a new branch.
    - Ensure all tests pass (`pytest`).
    - Update `CHANGELOG.md` with your changes.
    - Submit the PR.

## Code Standards

- Use **Black** for formatting.
- Use **isort** for imports.
- Include **type hints** for all new functions.
- Add tests for new functionality in the `tests/` directory.

## Reproducibility Priority

BioPAT is an academic benchmark. Any changes to the data processing pipeline must be strictly deterministic and documented.
