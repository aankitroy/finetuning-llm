from common.course import load_modules
from tests.conftest import REPO_ROOT


def test_loads_real_course():
    mods, warnings = load_modules(REPO_ROOT)
    assert warnings == []
    assert [m.id for m in mods] == [f"m{i:02d}" for i in range(17)]
    assert [m.order for m in mods] == list(range(17))
    assert all(m.next_finish.startswith("make ") for m in mods)


def test_broken_manifest_is_skipped_with_warning(copy_fixture):
    mods, warnings = load_modules(copy_fixture("course_basic"))
    assert [m.id for m in mods] == ["m00"]
    assert len(warnings) == 2
    assert "m01" in warnings[0]
    assert "m02" in warnings[1]


def test_metric_better_validated(tmp_path):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        'id = "m05"\ntitle = "T"\npart = "B"\norder = 5\nsummary = "s"\nest_minutes = 1\n'
        '[next]\nstart = "a"\nfinish = "make check M=05"\nwhy = "w"\n'
        '[[metrics]]\nkey = "k"\nlabel = "K"\nbetter = "sideways"\n'
    )
    mods, warnings = load_modules(tmp_path)
    assert mods == []
    assert len(warnings) == 1 and "better" in warnings[0]


def test_optional_sections_default_empty(copy_fixture):
    mods, _ = load_modules(copy_fixture("course_basic"))
    m = mods[0]
    assert m.artifacts == ["outputs/m00/*"]
    assert m.metrics == [{"key": "total_mem_gb", "label": "Total memory (GB)", "better": "higher"}]
    assert m.next_start.startswith("Read") and m.next_why == "Checks your environment"


def test_template_dir_not_loaded():
    mods, warnings = load_modules(REPO_ROOT)
    assert (REPO_ROOT / "modules" / "_template" / "module.toml").exists()
    assert all(m.path.name != "_template" for m in mods)
    assert not any("_template" in w for w in warnings)


def _write_manifest(tmp_path, extra: str):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        'id = "m05"\ntitle = "T"\npart = "B"\norder = 5\nsummary = "s"\nest_minutes = 1\n'
        "artifacts_placeholder = 0\n"
        '[next]\nstart = "a"\nfinish = "make check M=05"\nwhy = "w"\n' + extra
    )


def test_single_metrics_table_is_a_warning_not_a_crash(tmp_path):
    _write_manifest(tmp_path, '[metrics]\nkey = "k"\nlabel = "K"\nbetter = "lower"\n')
    mods, warnings = load_modules(tmp_path)
    assert mods == [] and len(warnings) == 1 and "metrics" in warnings[0]


def test_artifacts_as_list_is_a_warning_not_a_crash(tmp_path):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        'id = "m05"\ntitle = "T"\npart = "B"\norder = 5\nsummary = "s"\nest_minutes = 1\n'
        'artifacts = ["outputs/m05/*"]\n'
        '[next]\nstart = "a"\nfinish = "make check M=05"\nwhy = "w"\n'
    )
    mods, warnings = load_modules(tmp_path)
    assert mods == [] and len(warnings) == 1 and "artifacts" in warnings[0]


STEP_BASE = (
    'id = "m05"\ntitle = "T"\npart = "B"\norder = 5\nsummary = "s"\nest_minutes = 1\n'
    '[next]\nstart = "a"\nfinish = "make check M=05"\nwhy = "w"\n'
)


def test_steps_are_loaded(tmp_path):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        STEP_BASE + '[[steps]]\nkind = "lesson"\ntitle = "Run the lesson"\n'
        'detail = "d"\ncommand = "uv run jupyter lab x"\nfile = "lesson.ipynb"\n'
        '[[steps]]\nkind = "checkpoint"\ntitle = "Pass"\ndetail = "d"\ncommand = "make check M=05"\n'
    )
    (m,), warnings = load_modules(tmp_path)
    assert warnings == []
    assert [s["kind"] for s in m.steps] == ["lesson", "checkpoint"]
    assert m.steps[0]["file"] == "lesson.ipynb" and m.steps[1].get("file") is None


def test_unknown_step_kind_is_a_warning(tmp_path):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        STEP_BASE + '[[steps]]\nkind = "dance"\ntitle = "t"\ndetail = "d"\n'
    )
    mods, warnings = load_modules(tmp_path)
    assert mods == [] and "step" in warnings[0] and "dance" in warnings[0]


def test_steps_default_empty(copy_fixture):
    mods, _ = load_modules(copy_fixture("course_basic"))
    assert mods[0].steps == []
