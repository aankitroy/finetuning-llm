"""M7's train.py: a fair comparison, and plain messages for the mistakes a learner makes.

Uses tiny models only. Skipped where PyTorch (and, for the mlx parts, MLX) isn't installed.
"""

import importlib.util
from types import SimpleNamespace

import pytest

from tests.conftest import REPO_ROOT

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

M = REPO_ROOT / "modules" / "m07_lora_tools"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train():
    return _load(M / "train.py", "m07_train_under_test")


@pytest.fixture(scope="module")
def solutions():
    return _load(M / "solutions" / "exercises.py", "m07_solutions_under_test")


ARGS = SimpleNamespace(r=8, alpha=16.0, lr=2e-4, seed=0)
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _needs_mlx(train):
    if not train.HAS_MLX:
        pytest.skip("MLX needs an Apple-silicon Mac")


def test_mlx_optimizer_steps_like_torch_adamw(train):
    """Same gradients, same learning rate: both optimizers must move the weights the same."""
    _needs_mlx(train)
    import mlx.core as mx
    import numpy as np

    steps = 20
    opt = train.mlx_optimizer(ARGS, steps)
    w_torch = torch.nn.Parameter(torch.zeros(4))
    topt = torch.optim.AdamW([w_torch], lr=ARGS.lr, weight_decay=0.0)
    m5 = train.load_m05_train()
    warmup, total = train.lr_schedule(ARGS, steps)
    params = {"w": mx.zeros(4)}
    rng = np.random.default_rng(0)
    for step in range(5):
        g = rng.normal(size=4).astype(np.float32) * 0.1  # small: below the clipping norm
        for group in topt.param_groups:
            group["lr"] = m5.lr_at(step, ARGS.lr, warmup, total)
        w_torch.grad = torch.tensor(g)
        topt.step()
        params = opt.apply_gradients({"w": mx.array(g)}, params)
    np.testing.assert_allclose(np.array(params["w"]), w_torch.detach().numpy(), rtol=1e-4)


def _mlx_cfg(**lora):
    base = {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": [f"self_attn.{t}" for t in ATTN]}
    return {"num_layers": 2, "lora_parameters": base | lora}


@pytest.mark.parametrize(
    "cfg",
    [
        _mlx_cfg(keys=ATTN),  # plain names: mlx-lm would wrap nothing
        {
            "num_layers": 2,
            "lora_parameters": {"rank": 8, "scale": 2.0, "keys": ["self_attn.q_proj"]},
        },
        {"num_layers": None, "lora_parameters": _mlx_cfg()["lora_parameters"]},
        _mlx_cfg(scale=16.0),
    ],
    ids=["plain-keys", "no-dropout", "no-num-layers", "alpha-as-scale"],
)
def test_bad_mlx_config_gets_a_plain_message(train, cfg):
    with pytest.raises(train.ExerciseError, match="exercise 2"):
        train.check_mlx_config(cfg, ARGS)


def test_good_mlx_config_passes(train, solutions):
    train.check_mlx_config(solutions.mlx_lora_config(8, 16.0, ATTN, 28), ARGS)


def test_peft_target_that_matches_nothing_gets_a_plain_message(train):
    from peft import LoraConfig
    from tiny_models import make_tiny_hf_model

    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["attn.q_proj"], task_type="CAUSAL_LM")
    with pytest.raises(train.ExerciseError, match="exercise 3"):
        train.peft_wrap(make_tiny_hf_model(), cfg, ARGS)


def test_unwritten_exercises_stop_the_run_before_any_training(train):
    starter = _load(M / "starter" / "exercises.py", "m07_starter_under_test")
    with pytest.raises(NotImplementedError, match="Exercise 1"):
        train.preflight(starter)
    for name in ("tokens_per_sec", "load_peft_adapter", "load_mlx_adapter"):
        partly = _load(M / "solutions" / "exercises.py", f"m07_partly_{name}")

        def unwritten(*_, _name=name):
            raise NotImplementedError(f"Exercise: {_name}")

        setattr(partly, name, unwritten)
        with pytest.raises(NotImplementedError, match=name):
            train.preflight(partly)


def test_preflight_passes_with_the_solutions(train, solutions):
    train.preflight(solutions)


def test_a_tensor_from_tokens_per_sec_is_fine(train):
    assert train.as_speed(torch.tensor(5.0)) == 5.0
    with pytest.raises(train.ExerciseError, match="exercise 4"):
        train.as_speed("fast")


def test_m5_exercise_errors_get_a_plain_message(train, monkeypatch, capsys):
    m5 = train.load_m05_train()

    def boom(*_):
        raise m5.ExerciseError("collate should return a dict")

    monkeypatch.setattr(train, "run_tool", boom)
    assert train.run_one("peft", ARGS) == 1
    out = capsys.readouterr().out
    assert "collate should return a dict" in out and "Traceback" not in out


def test_not_yet_message_counts_six_exercises(train, monkeypatch, capsys):
    def unwritten(*_):
        raise NotImplementedError("Exercise 4: tokens_per_sec")

    monkeypatch.setattr(train, "run_tool", unwritten)
    assert train.run_one("peft", ARGS) == 1
    assert "all six exercises" in capsys.readouterr().out
