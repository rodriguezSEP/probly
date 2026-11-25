# How to use tabpfn in probly

## Installation

```sh
    uv sync --group tabpfn
```

Test if tabpfn is installed

```sh
    uv run python -c "import tabpfn; from importlib.metadata import version; print(version('tabpfn'))"
```

Should show >=10.0.0
