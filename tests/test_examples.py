import re
from pathlib import Path

import yaml

from tooltangle.sources import load_python_tools, read_config

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"


def test_readme_points_at_existing_examples():
    mentioned = set(re.findall(r"examples/[\w./-]+", (ROOT / "README.md").read_text()))
    assert mentioned
    missing = sorted(path for path in mentioned if not (ROOT / path.rstrip(".")).exists())
    assert not missing


def test_mcp_examples_are_valid_configs():
    configs = sorted(EXAMPLES.glob("*.json"))
    assert configs
    for path in configs:
        servers, skipped = read_config(path)
        assert servers and not skipped, path


def test_python_example_loads():
    specs = load_python_tools(f"{EXAMPLES / 'support_tools.py'}:TOOLS")
    assert len(specs) == 6
    assert all(spec.description for spec in specs)


def test_workflow_example_runs_a_check():
    workflow = yaml.safe_load((EXAMPLES / "github-workflow.yml").read_text())
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    assert any("tooltangle check" in step.get("run", "") for step in steps)
