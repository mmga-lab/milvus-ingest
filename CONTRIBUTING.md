# Contributing to milvus-ingest

Thank you for your interest in contributing to milvus-ingest! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/milvus-ingest.git
   cd milvus-ingest
   ```

2. **Set up Development Environment**
   ```bash
   # Install PDM if you haven't already
   pip install pdm

   # Install dependencies
   pdm install --dev
   ```

3. **Run Tests**
   ```bash
   pdm run pytest
   pdm run ruff check .
   pdm run mypy src/
   ```

## 📋 Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes
We use conventional commits for clear history:
```bash
git commit -m "feat: add new field type support"
git commit -m "fix: resolve validation edge case"
git commit -m "docs: update schema format examples"
```

### 4. Push and Create PR
```bash
git push origin feature/your-feature-name
```
Then create a pull request on GitHub.

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pdm run pytest

# Run with coverage
pdm run pytest --cov=src/milvus_fake_data --cov-report=term-missing

# Run specific test file
pdm run pytest tests/test_generator.py

# Run with verbose output
pdm run pytest -v
```

### Test Structure
- `tests/test_generator.py` - Core data generation tests
- `tests/test_cli.py` - CLI interface tests
- `tests/conftest.py` - Shared test fixtures

### Adding Tests
When adding new features:
1. Add unit tests for the core functionality
2. Add integration tests for CLI features
3. Include edge cases and error conditions
4. Ensure good test coverage

## 🔍 Code Quality

### Linting and Formatting
```bash
# Check code style
pdm run ruff check .

# Auto-fix issues
pdm run ruff check . --fix

# Format code
pdm run ruff format .

# Type checking
pdm run mypy src/
```

### Manual Quality Checks
Run quality checks manually before committing:
```bash
# Run all quality checks
pdm run pytest
pdm run ruff check . --fix
pdm run ruff format .
pdm run mypy src/
```

## 📝 Code Style

### Python Code Style
- Follow PEP 8 with line length of 88 characters
- Use type hints for all functions
- Write descriptive docstrings
- Prefer explicit over implicit code

### Example Code Style
```python
def generate_mock_data(
    schema_path: str | Path,
    rows: int = 1000,
    seed: int | None = None
) -> pd.DataFrame:
    """Generate mock data according to the schema.

    Args:
        schema_path: Path to JSON or YAML schema file.
        rows: Number of rows to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        A pandas DataFrame containing mock data.

    Raises:
        ValidationError: If schema validation fails.
        FileNotFoundError: If schema file doesn't exist.
    """
```

### Schema Design
- Use Pydantic models for validation
- Provide clear error messages with examples
- Support both dict and list schema formats
- Include comprehensive field validation

## 🐛 Reporting Issues

### Bug Reports
Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Schema file (if applicable)
- Full error message

### Feature Requests
Use the feature request template and include:
- Clear description of the feature
- Use case and motivation
- Example usage
- Potential implementation ideas

## 🏗️ Project Structure

```
milvus-ingest/
├── src/milvus_fake_data/
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── generator.py    # Core data generation logic
│   ├── models.py       # Pydantic validation models
│   └── exceptions.py   # Custom exceptions
├── tests/
│   ├── test_cli.py
│   ├── test_generator.py
│   └── conftest.py
├── .github/
│   └── workflows/      # CI/CD workflows
├── pyproject.toml      # Project configuration
└── README.md
```

## 🔄 CI/CD Pipeline

Our GitHub Actions workflows automatically:

### On Pull Requests and Pushes
- **Testing**: Run tests on Python 3.10, 3.11, 3.12 across Ubuntu, Windows, macOS
- **Linting**: Check code style with ruff and type checking with mypy
- **Security**: Scan for vulnerabilities and security issues
- **Build**: Verify package builds correctly

### On Release Tags
- **Build and Test**: Full test suite and package verification
- **Publish**: Automatic publishing to PyPI
- **Docker**: Build and push container images
- **GitHub Releases**: Create release with artifacts

### Continuous Monitoring
- **Dependencies**: Automated dependency updates via Dependabot
- **Security**: Weekly security scans
- **Project Management**: Auto-labeling and issue triage

## 🚢 Release Process

### Version Bumping
1. Update version in `pyproject.toml`
2. Create a git tag: `git tag v0.2.0`
3. Push tag: `git push origin v0.2.0`
4. GitHub Actions will handle the rest!

### Release Types
- **Patch** (0.1.1): Bug fixes
- **Minor** (0.2.0): New features, backward compatible
- **Major** (1.0.0): Breaking changes

## 💬 Getting Help

- **Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🎉 Recognition

Contributors are automatically recognized in:
- GitHub contributors graph
- Release notes
- Special thanks in major releases

Thank you for contributing to milvus-ingest! 🚀
