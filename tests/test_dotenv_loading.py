"""Tests for .env loading from CWD.

Tests:
- load_dotenv_from_cwd() loads missing vars
- load_dotenv_from_cwd() does NOT override existing vars
- load_dotenv_from_cwd() is silent when .env missing
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

# Ensure src directory is in path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def test_dotenv_loads_missing_vars(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CWD .env should fill missing environment variables."""
    # Create .env file in temp directory
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_VAR=from-dotenv\n")

    # Switch to temp directory
    monkeypatch.chdir(tmp_path)

    # Ensure the variable is not set
    monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)

    # Import and call the loader (must be done AFTER chdir)
    # Use importlib to force re-import since module-level load already happened
    from lazydspy.cli import load_dotenv_from_cwd

    load_dotenv_from_cwd()

    assert os.environ.get("TEST_DOTENV_VAR") == "from-dotenv"


def test_dotenv_does_not_override_existing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CWD .env should NOT override existing environment variables."""
    # Create .env file with a value
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_OVERRIDE=from-dotenv\n")

    # Switch to temp directory
    monkeypatch.chdir(tmp_path)

    # Set the variable BEFORE loading
    monkeypatch.setenv("TEST_DOTENV_OVERRIDE", "from-process")

    # Import and call the loader
    from lazydspy.cli import load_dotenv_from_cwd

    load_dotenv_from_cwd()

    # Should still be the process value, not overwritten
    assert os.environ.get("TEST_DOTENV_OVERRIDE") == "from-process"


def test_dotenv_missing_file_silent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing .env file should not raise errors."""
    # Switch to temp directory (no .env file)
    monkeypatch.chdir(tmp_path)

    # Import and call the loader - should not raise
    from lazydspy.cli import load_dotenv_from_cwd

    load_dotenv_from_cwd()  # Should complete without exception


def test_dotenv_unreadable_file_silent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable .env file should not raise errors (silent failure)."""
    # Create .env file
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=value\n")

    # Switch to temp directory
    monkeypatch.chdir(tmp_path)

    # Make file unreadable (platform-specific, may not work on all systems)
    try:
        env_file.chmod(0o000)
        file_made_unreadable = True
    except (OSError, PermissionError):
        # Skip permission test on platforms that don't support it
        file_made_unreadable = False

    if file_made_unreadable:
        try:
            # Import and call the loader - should not raise
            from lazydspy.cli import load_dotenv_from_cwd

            load_dotenv_from_cwd()  # Should complete without exception
        finally:
            # Restore permissions for cleanup
            env_file.chmod(0o644)


def test_dotenv_with_comments(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CWD .env with comments should be parsed correctly."""
    # Create .env file with comments
    env_content = """# This is a comment
TEST_DOTENV_COMMENT=value-after-comment
# Another comment
TEST_DOTENV_COMMENT2=second-value
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    # Switch to temp directory
    monkeypatch.chdir(tmp_path)

    # Ensure variables are not set
    monkeypatch.delenv("TEST_DOTENV_COMMENT", raising=False)
    monkeypatch.delenv("TEST_DOTENV_COMMENT2", raising=False)

    # Import and call the loader
    from lazydspy.cli import load_dotenv_from_cwd

    load_dotenv_from_cwd()

    assert os.environ.get("TEST_DOTENV_COMMENT") == "value-after-comment"
    assert os.environ.get("TEST_DOTENV_COMMENT2") == "second-value"
