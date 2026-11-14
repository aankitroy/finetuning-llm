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
