"""Verify that every library the project needs is installed.

Usage:
    python src/check_env.py
"""
import importlib
import platform
import sys

REQUIRED_PACKAGES = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "pyarrow",
                     "yaml", "requests", "joblib", "streamlit", "pytest"]
OPTIONAL_PACKAGES = ["shap"]


def _version(package: str) -> str | None:
    try:
        module = importlib.import_module(package)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {platform.machine()}")

    missing = []
    for package in REQUIRED_PACKAGES:
        version = _version(package)
        if version is None:
            print(f"  MISSING  {package}")
            missing.append(package)
        else:
            print(f"  OK       {package:<12} {version}")

    for package in OPTIONAL_PACKAGES:
        version = _version(package)
        status = f"OK       {package:<12} {version}" if version else \
            f"OPTIONAL {package:<12} not installed (explanations use the built-in method)"
        print(f"  {status}")

    if missing:
        print("\nRun: pip install -r requirements.txt")
        return 1

    print("\nEnvironment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
