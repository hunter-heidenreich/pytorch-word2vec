#!/usr/bin/env python3
"""Simple test runner script for development."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the test suite."""
    # Change to project root
    project_root = Path(__file__).parent

    # Install test dependencies if needed
    print("Installing test dependencies...")
    subprocess.run(["uv", "sync", "--extra", "test"], cwd=project_root, check=True)

    # Run tests
    print("\nRunning tests...")
    result = subprocess.run(["uv", "run", "pytest", "tests/", "-v"], cwd=project_root)

    return result.returncode


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    project_root = Path(__file__).parent

    print("Running tests with coverage...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/",
            "-v",
            "--cov=src/modern_word2vec",
            "--cov-report=term-missing",
            "--cov-report=html",
        ],
        cwd=project_root,
    )

    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--coverage":
        exit_code = run_tests_with_coverage()
    else:
        exit_code = run_tests()

    sys.exit(exit_code)
