"""The six run configs are the experiment's fairness claim in machine-checkable form.

The assignment requires the three arms to share model, task order, in-problem evolution rounds,
tools and generation parameters. That is easy to write in a README and easy to break with one
stray edit six weeks later. These tests read the configs the way the driver does -- through
upstream's ``load_run_configuration``, so the ``base_config:`` merge is exercised rather than
assumed -- and fail if an arm ever drifts.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from alphaapollo.core.generation.evolving.utils.utils import load_run_configuration

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "configs"
ARMS = ("baseline", "raw", "evo")
PHASES = ("adapt", "heldout")
ALL_RUNS = [f"{phase}_{arm}" for phase in PHASES for arm in ARMS]


def load(name: str) -> dict:
    bundle = load_run_configuration(str(CONFIG_DIR / f"harness_{name}.yaml"))
    return OmegaConf.to_container(bundle["cfg"], resolve=True)


def solver_environment(cfg: dict) -> dict:
    """Everything the three arms must hold identical. Excludes run_dir/tag/wandb naming, which
    differ by construction, and the arm block itself, which is the manipulation."""
    env = cfg["env"]["config"]["informal_math_evolving"]
    harness = cfg["harness"]
    return {
        "evolving_round": env["evolving_round"],
        "enable_verify": env["enable_verify"],
        "policy_env": env["policy_env"],
        "verifier_env": env["verifier_env"],
        "verifier_max_workers": env["concurrency"]["verifier_max_workers"],
        "verifier_env_num": cfg["run"]["verifier_env_num"],
        "policy_env_num": cfg["run"]["policy_env_num"],
        "policy_model_cfg": cfg["policy_model_cfg"],
        "verifier_cfg": cfg["verifier_cfg"],
        "batch_size": harness["batch_size"],
        "seed": harness["seed"],
        "budget": harness["budget"],
        "feedback_level": harness["feedback_level"],
    }


@pytest.mark.parametrize("name", ALL_RUNS)
def test_every_config_loads_through_the_drivers_own_loader(name):
    assert load(name)["harness"]["arm"] in ARMS


def test_all_six_runs_share_one_solver_environment():
    """The single most load-bearing assertion here: if this fails, the three arms are no longer
    a controlled comparison and no result from them means anything."""
    environments = {name: solver_environment(load(name)) for name in ALL_RUNS}
    reference = environments["adapt_baseline"]
    for name, env in environments.items():
        assert env == reference, f"{name} drifted from the shared solver environment"


def test_the_raw_and_evo_arms_receive_the_same_injection_budget():
    """The experiment asks whether compiled skills beat raw history. If one arm could inject
    more text than the other, it would instead be measuring context length."""
    assert load("adapt_raw")["harness"]["budget"] == load("adapt_evo")["harness"]["budget"]


@pytest.mark.parametrize("name", ALL_RUNS)
def test_no_config_contains_an_api_key(name):
    """The assignment forbids committing keys; every model block reads OPENAI_API_KEY instead."""
    cfg = load(name)
    for block in ("policy_model_cfg", "verifier_cfg", "vllm_config"):
        assert cfg[block]["api_key"] == "", f"{name}.{block} must keep api_key empty"
    selector = cfg["harness"].get("selector_model_cfg")
    if selector is not None:
        assert selector["api_key"] == ""


@pytest.mark.parametrize("name", ALL_RUNS)
def test_raw_text_of_every_config_is_free_of_key_material(name):
    assert "sk-" not in (CONFIG_DIR / f"harness_{name}.yaml").read_text()


@pytest.mark.parametrize("arm", ARMS)
def test_adaptation_runs_are_not_frozen_and_heldout_runs_are(arm):
    assert load(f"adapt_{arm}")["harness"]["frozen"] is False
    assert load(f"heldout_{arm}")["harness"]["frozen"] is True


@pytest.mark.parametrize("arm", ARMS)
def test_the_two_phases_read_different_streams(arm):
    adapt = load(f"adapt_{arm}")["harness"]["stream_path"]
    heldout = load(f"heldout_{arm}")["harness"]["stream_path"]
    assert "adaptation" in adapt and "heldout" in heldout


@pytest.mark.parametrize("arm", ("raw", "evo"))
def test_a_heldout_run_never_points_at_the_adaptation_runs_own_store(arm):
    """A frozen run must not be able to append to the artifact it is evaluating. The run script
    copies the store; these paths are what make that copy mandatory rather than optional."""
    adapt = load(f"adapt_{arm}")["harness"]["store_root"]
    heldout = load(f"heldout_{arm}")["harness"]["store_root"]
    assert adapt != heldout


def test_only_the_evo_arm_declares_growth_caps():
    """Caps bound harness growth. Baseline has nothing to grow; Raw's pool is deliberately
    uncurated -- giving it caps would quietly make it a weak Evo arm instead of its own thing."""
    assert "caps" in load("adapt_evo")["harness"]
    assert "caps" not in load("adapt_baseline")["harness"]
    assert "caps" not in load("adapt_raw")["harness"]


def test_baseline_declares_no_store_so_it_cannot_accumulate_anything():
    for phase in PHASES:
        assert "store_root" not in load(f"{phase}_baseline")["harness"]


@pytest.mark.parametrize("name", ALL_RUNS)
def test_every_run_writes_into_its_own_directory(name):
    """Two runs sharing a run_dir would interleave metrics.jsonl and corrupt each other's
    resume marker."""
    run_dirs = {n: load(n)["harness"]["run_dir"] for n in ALL_RUNS}
    assert len(set(run_dirs.values())) == len(ALL_RUNS)


@pytest.mark.parametrize("name", ALL_RUNS)
def test_every_run_dir_is_inside_the_repository_not_a_temp_directory(name):
    """Diagnostic runs wrote to the session scratchpad, which the OS reclaims. A 15-hour run
    must not."""
    run_dir = load(name)["harness"]["run_dir"]
    assert run_dir.startswith("./outputs/"), run_dir


@pytest.mark.parametrize("name", ALL_RUNS)
def test_every_config_routes_to_the_harness_driver_not_upstreams(name):
    """Read through ``api.load_config`` -- the launcher's own reader -- and not through
    ``load_run_configuration`` like the rest of this file.

    These are two different readers with two different behaviours, and the routing decision
    belongs to the first one. ``api.evo`` picks the entrypoint with a plain ``OmegaConf.load``
    that never applies ``base_config`` (workflows/api.py:239), while the driver's loader does
    merge it. An earlier version of this test used the merging reader, passed, and hid the fact
    that every single run was launching upstream's ``evolving_main`` instead -- the same class of
    failure as the ``--config_path`` flag that once sent a whole diagnostic to upstream and
    reported "Finished run" after zero model calls.

    Asserting through the wrong reader is worse than not asserting: it converts a loud crash
    into a green test.
    """
    from alphaapollo.workflows.api import load_config

    cfg = load_config(str(CONFIG_DIR / f"harness_{name}.yaml"))
    assert cfg.get("entrypoint_module", "").endswith("evolving_harness_main")
