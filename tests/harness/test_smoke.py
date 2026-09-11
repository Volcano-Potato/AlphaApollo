def test_harness_package_importable():
    import alphaapollo.core.harness as h
    assert h.__name__ == "alphaapollo.core.harness"


def test_harness_does_not_import_inproblem_memory():
    """概念边界的结构性保证：harness 包不得依赖题内 memory。"""
    import importlib
    import pkgutil

    import alphaapollo.core.harness as h

    offenders = []
    for mod in pkgutil.iter_modules(h.__path__):
        m = importlib.import_module(f"alphaapollo.core.harness.{mod.name}")
        src = open(m.__file__, encoding="utf-8").read()
        if "core.environments.memory" in src:
            offenders.append(mod.name)
    assert offenders == [], f"harness modules import in-problem memory: {offenders}"
