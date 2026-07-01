# Contributing to DRL Scheduler Statistical Testbed

Thank you for your interest in contributing! This project is part of an Honours research thesis, but contributions are welcome for:

- Bug fixes
- Documentation improvements
- Additional algorithm implementations
- Statistical analysis enhancements
- Reproducibility improvements

---

## Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top right of the GitHub page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/<repo>.git
cd <repo>
```

### 3. Set Up Development Environment

**Using Nix (recommended):**
```bash
nix develop
```

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Code Style

### Python

- **Formatter:** Black (88-char line limit)
- **Linter:** Flake8
- **Type hints:** Optional but encouraged for new code
- **Docstrings:** Google style

**Format your code:**
```bash
black src/
flake8 src/ --max-line-length=88
```

### Naming Conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Files | `snake_case.py` | `train_agent.py` |
| Classes | `PascalCase` | `DRLScheduler` |
| Functions/Methods | `snake_case` | `compute_metrics()` |
| Variables | `snake_case` | `waiting_time` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_EPISODES` |

---

## Testing

Before submitting a pull request:

1. **Run existing tests:**
    ```bash
    python -m pytest src/test_scheduler.py
    ```

2. **Add tests for new features** alongside the code in `src/` (e.g. `src/test_your_feature.py`).

3. **Smoke test your changes** end-to-end:
    ```bash
    just run_smoke
    # or a single run:
    python -m src.train_agents --algorithm maskable_ppo --name contrib_smoke \
      --trace data/splits/physical_job_dev70.tsv --save_interval 100 --total_saving 2 --seed 123456
    ```

---

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include parameter types and return types
- Provide usage examples for complex functions

**Example:**
```python
def compute_waiting_time(jobs: list[Job], completed_jobs: list[Job]) -> float:
    """
    Compute average waiting time for completed jobs.
    
    Args:
        jobs: List of all jobs in the system
        completed_jobs: List of jobs that have completed execution
        
    Returns:
        Average waiting time in seconds
        
    Example:
        >>> avg_wait = compute_waiting_time(all_jobs, finished_jobs)
        >>> print(f"Average wait: {avg_wait:.2f}s")
    """
    ...
```

### README Updates

If your contribution adds new features, update the main README:
- Add to "Features" section
- Update usage examples
- Add to table of contents if needed

---

## Pull Request Process

### 1. Commit Your Changes

```bash
git add .
git commit -m "Add feature: brief description"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- First line: <50 chars summary
- Body: Detailed explanation (if needed)

**Examples:**
```
Add MaskableA2C algorithm implementation

Implements masked action support for A2C algorithm following
the sb3-contrib MaskablePPO pattern. Includes tests and config.
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Open Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Added tests for new features
- [ ] All existing tests pass
- [ ] Smoke tested manually

## Checklist
- [ ] Code follows project style (Black + Flake8)
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 4. Code Review

- Respond to reviewer comments
- Make requested changes in new commits
- Push updates to the same branch

---

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify you're using the latest version
3. Test with a clean environment

### Bug Report Template

```markdown
**Description:**
Clear description of the bug

**To Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.12.1]
- PyTorch version: [e.g., 2.7.0]

**Additional Context:**
Logs, screenshots, etc.
```

---

## Feature Requests

Use the "Feature Request" issue template:

```markdown
**Problem Statement:**
What problem does this solve?

**Proposed Solution:**
How should it work?

**Alternatives Considered:**
Other approaches you've thought about

**Additional Context:**
Mockups, references, etc.
```

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Acknowledgments

Contributors will be acknowledged in:
- `README.md` (Contributors section)
- Thesis acknowledgments (for significant contributions)
- Future publications (if applicable)

---

## Questions?

- **Email:** [To be filled]
- **GitHub Issues:** Open an issue or discussion
- **Pull Request Comments:** Tag a reviewer

---

**Thank you for contributing to reproducible HPC scheduling research!**
