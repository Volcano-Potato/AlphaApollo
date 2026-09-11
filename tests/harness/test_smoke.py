import ast
from pathlib import Path

TARGET_PACKAGE = "alphaapollo.core.environments.memory"
TARGET_PARTS = tuple(TARGET_PACKAGE.split("."))


def _dotted_name_hits_target(name: str) -> bool:
    """True if the dotted module path `name` is the target package or one of its submodules."""
    parts = tuple(name.split("."))
    return parts[: len(TARGET_PARTS)] == TARGET_PARTS


def _imports_inproblem_memory(source: str) -> bool:
    """Return True iff `source` contains a real Python import of the in-problem memory package
    (or one of its submodules/members), based on the AST rather than a substring search.

    This deliberately ignores comments, docstrings, and ordinary string literals that merely
    *mention* the package name (e.g. to document the architectural boundary) -- only actual
    `import` / `from ... import ...` statements count as violations.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # e.g. `import alphaapollo.core.environments.memory` or `... as m`
                if _dotted_name_hits_target(alias.name):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                # relative `from . import x` inside this package can't reach the target.
                continue
            # `from alphaapollo.core.environments.memory import SimpleMemory`
            # `from alphaapollo.core.environments.memory.base import BaseMemory`
            if _dotted_name_hits_target(node.module):
                return True
            # `from alphaapollo.core.environments import memory`
            module_parts = tuple(node.module.split("."))
            if module_parts == TARGET_PARTS[:-1]:
                for alias in node.names:
                    if alias.name == TARGET_PARTS[-1]:
                        return True
    return False


def test_harness_package_importable():
    import alphaapollo.core.harness as h

    assert h.__name__ == "alphaapollo.core.harness"


def test_harness_does_not_import_inproblem_memory():
    """概念边界的结构性保证：harness 包不得依赖题内 memory。

    Scans every *.py file directly under the harness package directory -- including
    __init__.py, which pkgutil.iter_modules() never yields -- and checks each one with the
    AST-based `_imports_inproblem_memory`, so a mere textual mention (e.g. in a docstring
    explaining this very boundary) cannot trigger a false positive.
    """
    import alphaapollo.core.harness as h

    package_dir = Path(h.__file__).parent
    offenders = []
    for py_file in sorted(package_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        if _imports_inproblem_memory(src):
            offenders.append(py_file.name)
    assert offenders == [], f"harness modules import in-problem memory: {offenders}"


# --- Unit tests for the detector itself -------------------------------------------------
# The guard is only as trustworthy as this function; these cases pin down both failure modes
# that motivated the rewrite (false negative on __init__.py, false positive on substring match).

VIOLATION_SOURCES = {
    "plain import": "import alphaapollo.core.environments.memory\n",
    "plain import with alias": "import alphaapollo.core.environments.memory as m\n",
    "from-import of a name": "from alphaapollo.core.environments.memory import SimpleMemory\n",
    "from-import of the submodule itself": "from alphaapollo.core.environments import memory\n",
    "from-import reaching into a submodule": "from alphaapollo.core.environments.memory.base import BaseMemory\n",
}

SAFE_SOURCES = {
    "mention in a comment": "# see alphaapollo.core.environments.memory for the in-problem analogue\nx = 1\n",
    "mention in a module docstring": '"""This package must never import alphaapollo.core.environments.memory."""\nx = 1\n',
    "mention in a plain string literal": 'x = "alphaapollo.core.environments.memory"\n',
    "import of a sibling environments subpackage": "from alphaapollo.core.environments.prompts import x\n",
    "import of an unrelated stdlib module": "import os\nfrom pathlib import Path\n",
}


def test_detector_flags_every_known_violation_form():
    for label, src in VIOLATION_SOURCES.items():
        assert _imports_inproblem_memory(src), f"expected violation to be detected: {label!r}\nsource:\n{src}"


def test_detector_does_not_flag_mentions_or_unrelated_imports():
    for label, src in SAFE_SOURCES.items():
        assert not _imports_inproblem_memory(src), f"expected no violation for: {label!r}\nsource:\n{src}"
