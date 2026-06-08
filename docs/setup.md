# Environment Setup

## Nix (Recommended)

The Nix development environment provides a fully reproducible setup with pinned dependencies.

### Installation

1. **Install Nix** (if not already installed):
   ```bash
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. **Enable flakes** (add to `~/.config/nix/nix.conf`):
   ```
   experimental-features = nix-command flakes
   ```

3. **Enter development environment**:
   ```bash
   cd HPC-DRL-Scheduler
   nix develop
   ```

### Pinned Versions (from flake.nix)

- **Python:** 3.12
- **PyTorch:** 2.7.0
- **Stable-Baselines3:** 2.6.0
- **Gymnasium:** 1.1.0
- **NumPy:** 1.26.4
- **SciPy:** 1.14.1

---

## pip (Fallback)

If you cannot use Nix, install dependencies via pip. This path is not guaranteed to be fully reproducible and should be treated as best-effort.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch==2.7.0 stable-baselines3==2.6.0 sb3-contrib==2.4.0 gymnasium==1.1.0 numpy==1.26.4 scipy==1.14.1 matplotlib==3.9.0 pandas==2.2.0 pyyaml==6.0.1
```

## Verification

Test your installation:

```bash
python -c "import torch; import stable_baselines3; import gymnasium; print('✅ All imports successful')"
```

Expected output:
```
✅ All imports successful
```

---

## GPU Support

For GPU-accelerated training, ensure CUDA is installed:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If `False`, reinstall PyTorch with CUDA support:
```bash
pip install torch==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sb3_contrib'`

**Solution:**
```bash
pip install sb3-contrib==2.4.0
```

### Issue: Nix build fails with "unfree package"

**Solution:** Allow unfree packages in `~/.config/nixpkgs/config.nix`:
```nix
{ allowUnfree = true; }
```

---

## Development Tools (Optional)

For code quality and formatting:

```bash
pip install black flake8 pytest
```

Format code:
```bash
black training/ evaluation/ statistical_analysis/
```

Lint:
```bash
flake8 training/ evaluation/ statistical_analysis/ --max-line-length=88
```
