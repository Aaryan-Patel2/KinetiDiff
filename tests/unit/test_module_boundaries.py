"""Verify that src/kinetidiff/gcdm/ does not import guidance or affinity_pred.

The gcdm sub-package must remain self-contained. Cross-package dependencies
(gcdm → guidance, gcdm → affinity_pred) would create circular import risk
and couple the core diffusion model to higher-level orchestration code.
"""
import ast
from pathlib import Path

GCDM_ROOT = Path(__file__).resolve().parents[2] / "src" / "kinetidiff" / "gcdm"

FORBIDDEN_IMPORTS = [
    "kinetidiff.guidance",
    "kinetidiff.affinity_pred",
]


def _get_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _collect_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]


def test_gcdm_does_not_import_guidance():
    violations = []
    for py_file in _collect_py_files(GCDM_ROOT):
        source = py_file.read_text(encoding="utf-8")
        for imp in _get_imports(source):
            if imp.startswith("kinetidiff.guidance"):
                violations.append(f"{py_file.relative_to(GCDM_ROOT)}: imports {imp!r}")
    assert not violations, "gcdm must not import kinetidiff.guidance:\n" + "\n".join(violations)


def test_gcdm_does_not_import_affinity_pred():
    violations = []
    for py_file in _collect_py_files(GCDM_ROOT):
        source = py_file.read_text(encoding="utf-8")
        for imp in _get_imports(source):
            if imp.startswith("kinetidiff.affinity_pred"):
                violations.append(f"{py_file.relative_to(GCDM_ROOT)}: imports {imp!r}")
    assert not violations, (
        "gcdm must not import kinetidiff.affinity_pred:\n" + "\n".join(violations)
    )
