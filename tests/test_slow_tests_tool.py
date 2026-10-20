"""`tools/slow_tests.py`: the slow test modules CI's fast set leaves out, read from a timed run's JUnit file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "slow_tests.py"
spec = importlib.util.spec_from_file_location("slow_tests", TOOL)
slow_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slow_tests)

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_a" name="test_one" time="2.5"/>
  <testcase classname="tests.test_a" name="test_two" time="0.6"/>
  <testcase classname="tests.test_b.TestGroup" name="test_three[x]" time="1.0"/>
  <testcase classname="tests.test_b.TestGroup" name="test_three[y]" time="1.0"/>
  <testcase classname="tests.test_c" name="test_four" time="2.9"/>
</testsuite></testsuites>
"""


def test_a_module_is_slow_by_the_sum_of_its_tests_and_a_class_belongs_to_its_module(tmp_path):
    """Contract: a module's time is the sum over its test cases, a test class's cases count to its module, and a module
    at or over the threshold is slow. By hand: test_a 2.5 + 0.6 = 3.1 s (slow at 3), test_b 1.0 + 1.0 = 2.0 s (fast),
    test_c 2.9 s (fast at 3, slow at 2). Regression caught: per-test thresholds, which keep a module whose fixture cost
    sits in one test, or a class name read as the module."""
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT, encoding="utf-8")
    times = slow_tests.module_times(junit)
    assert times == {"test_a": 3.1, "test_b": 2.0, "test_c": 2.9}
    assert slow_tests.slow_modules(times, 3.0) == ["test_a"]
    assert slow_tests.slow_modules(times, 2.0) == ["test_a", "test_b", "test_c"]


GROUPED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_build" name="test_builds@scripts" time="2.8"/>
  <testcase classname="tests.test_reads" name="test_reads_one@scripts" time="0.1"/>
  <testcase classname="tests.test_reads" name="test_reads_two[x]@scripts" time="0.2"/>
  <testcase classname="tests.test_other" name="test_other@other" time="1.0"/>
  <testcase classname="tests.test_plain" name="test_plain" time="0.5"/>
</testsuite></testsuites>
"""


def test_the_modules_that_share_an_xdist_group_are_slow_together(tmp_path):
    """Contract: the modules whose tests share an xdist group (the `@group` that `--dist loadgroup` ends a test's name
    with) are summed as one unit and are slow together, and a module in no shared group is its own unit. By hand:
    the group `scripts` is test_build 2.8 s + test_reads 0.1 + 0.2 s = 3.1 s, slow at 3 though neither module is;
    test_other (1.0 s, its own group) and test_plain (0.5 s, no group) are fast. Regression caught: a group's shared
    build charged to the module that ran first, so the fast set keeps the other module and pays the build there (the
    AMR's docking tests, 13 s of setup once `test_job_amr` was left out)."""
    junit = tmp_path / "junit.xml"
    junit.write_text(GROUPED, encoding="utf-8")
    times, groups = slow_tests.module_times(junit), slow_tests.module_groups(junit)
    assert groups == {"scripts": {"test_build", "test_reads"}, "other": {"test_other"}}
    assert slow_tests.slow_modules(times, 3.0) == []
    assert slow_tests.slow_modules(times, 3.0, groups) == ["test_build", "test_reads"]
