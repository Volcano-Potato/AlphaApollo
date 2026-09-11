import ast
from pathlib import Path

TARGET_PACKAGE = "alphaapollo.core.environments.memory"
TARGET_PARTS = tuple(TARGET_PACKAGE.split("."))

# The guard only ever scans files that live directly under alphaapollo/core/harness/ (a flat,
# non-nested package -- see test_harness_does_not_import_inproblem_memory below), so relative
# imports found in those files always resolve against this fixed package as their base.
HARNESS_PACKAGE = "alphaapollo.core.harness"
HARNESS_PARTS = tuple(HARNESS_PACKAGE.split("."))


def _dotted_name_hits_target(name: str) -> bool:
    """True if the dotted module path `name` is the target package or one of its submodules."""
    parts = tuple(name.split("."))
    return parts[: len(TARGET_PARTS)] == TARGET_PARTS


def _relative_import_base(level: int) -> tuple[str, ...] | None:
    """Resolve a relative-import `level` (the number of leading dots) to the absolute package
    it's relative to, given that the importing module lives directly inside HARNESS_PACKAGE.

    level=1 (`from . import x` / `from .x import y`) -> HARNESS_PACKAGE itself.
    level=2 (`from .. import x`)                      -> its parent package.
    level=3 (`from ... import x`)                     -> its grandparent package.

    Returns None if `level` escapes above the top of HARNESS_PARTS (e.g. level=4 here) -- there
    is nothing sensible to compare against, so the caller should treat it as "no match" rather
    than raise or guess.
    """
    if level <= 0 or level > len(HARNESS_PARTS):
        return None
    return HARNESS_PARTS[: len(HARNESS_PARTS) - level + 1]


def _imports_inproblem_memory(source: str) -> bool:
    """Return True iff `source` contains a real Python import of the in-problem memory package
    (or one of its submodules/members), based on the AST rather than a substring search.

    This deliberately ignores comments, docstrings, and ordinary string literals that merely
    *mention* the package name (e.g. to document the architectural boundary) -- only actual
    `import` / `from ... import ...` statements count as violations. Both absolute imports and
    relative imports (`from .`, `from ..`, `from ...`) are resolved and checked; relative imports
    are resolved against HARNESS_PACKAGE per `_relative_import_base`.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # e.g. `import alphaapollo.core.environments.memory` or `... as m`
                if _dotted_name_hits_target(alias.name):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import: `from . import x`, `from ..environments.memory import y`, ...
                base = _relative_import_base(node.level)
                if base is None:
                    continue
                module_parts = base + tuple(node.module.split(".")) if node.module else base
            else:
                if node.module is None:
                    continue
                module_parts = tuple(node.module.split("."))

            # `from alphaapollo.core.environments.memory import SimpleMemory`
            # `from alphaapollo.core.environments.memory.base import BaseMemory`
            # `from ..environments.memory import SimpleMemory` (resolved the same way)
            if module_parts and _dotted_name_hits_target(".".join(module_parts)):
                return True
            # `from alphaapollo.core.environments import memory`
            # `from ...core.environments import memory` (resolved the same way)
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
    "relative from-import reaching into a submodule (two dots)": "from ..environments.memory import SimpleMemory\n",
    "relative from-import of the submodule itself (three dots)": "from ...core.environments import memory\n",
    "import inside a function body": "def f():\n    import alphaapollo.core.environments.memory\n",
}

SAFE_SOURCES = {
    "mention in a comment": "# see alphaapollo.core.environments.memory for the in-problem analogue\nx = 1\n",
    "mention in a module docstring": '"""This package must never import alphaapollo.core.environments.memory."""\nx = 1\n',
    "mention in a plain string literal": 'x = "alphaapollo.core.environments.memory"\n',
    "import of a sibling environments subpackage": "from alphaapollo.core.environments.prompts import x\n",
    "import of an unrelated stdlib module": "import os\nfrom pathlib import Path\n",
    "import of a differently-named package with a matching prefix": "import alphaapollo.core.environments.memoryX\n",
    "relative import within the harness package itself": "from . import schema\n",
}


def test_detector_flags_every_known_violation_form():
    for label, src in VIOLATION_SOURCES.items():
        assert _imports_inproblem_memory(src), f"expected violation to be detected: {label!r}\nsource:\n{src}"


def test_detector_does_not_flag_mentions_or_unrelated_imports():
    for label, src in SAFE_SOURCES.items():
        assert not _imports_inproblem_memory(src), f"expected no violation for: {label!r}\nsource:\n{src}"
