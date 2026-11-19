"""Repository conventions that the project enforces with tests.

These are not style preferences: each one is a rule the interface or the docs must
obey, and a regression would be invisible in a normal unit test.
"""

from __future__ import annotations

import pathlib
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json"}
SKIP_PARTS = {".venv", ".git", "models", "database", "__pycache__", "benchmarks"}


def text_files() -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name == ".gitignore":
            found.append(path)
    return found


class CopyRuleTests(unittest.TestCase):
    def test_no_em_dash_anywhere(self) -> None:
        """The em dash reads as machine-written copy and is banned outright."""
        offenders = []
        for path in text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "\u2014" in text:
                relative = path.relative_to(PROJECT_ROOT)
                offenders.append(f"{relative}:{text.count(chr(0x2014))}")
        self.assertEqual(offenders, [], "em dash found in: " + ", ".join(offenders))


class RepositoryHygieneTests(unittest.TestCase):
    def test_gitignore_excludes_biometric_data(self) -> None:
        ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("database/", "models/", "settings.json", ".venv/"):
            self.assertIn(pattern, ignore)

    def test_readme_documents_the_limitations(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertIn("not certification", readme)
        self.assertIn("not a probability", readme)

    def test_the_technical_document_is_linked_from_the_readme(self) -> None:
        """The README stays short, so the depth lives in a file it must point at."""
        technical = PROJECT_ROOT / "TECHNICAL.md"
        self.assertTrue(technical.is_file(), "TECHNICAL.md is missing")
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("TECHNICAL.md", readme)
        self.assertGreater(len(technical.read_text(encoding="utf-8")), 5000)

    def test_the_readme_stays_short(self) -> None:
        readme = PROJECT_ROOT / "README.md"
        lines = readme.read_text(encoding="utf-8").splitlines()
        self.assertLess(len(lines), 250, "the README is meant to stay a short overview")

    def test_every_source_module_has_a_docstring(self) -> None:
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            if path.name == "__init__.py":
                continue
            text = path.read_text(encoding="utf-8")
            self.assertTrue(
                text.startswith(('"""', "from __future__", '"""')),
                f"{path.name} does not start with a module docstring",
            )
            self.assertIn('"""', text[:400], f"{path.name} has no module docstring")


if __name__ == "__main__":
    unittest.main()
