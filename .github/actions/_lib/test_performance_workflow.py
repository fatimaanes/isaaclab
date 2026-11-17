# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise workflow activation, source mounts, and cancellation without starting GPU jobs."""

import ast
import json
import os
import signal
import subprocess
import sys
import textwrap
from contextlib import suppress
from inspect import signature
from pathlib import Path

import pytest
import yaml

from tools.performance_regression.source_revision import MEASUREMENT_PATHS
from tools.performance_regression.metrics import extract

_WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/build.yaml"


def _labeled(login: str) -> dict[str, str | dict[str, str]]:
    return {"event": "labeled", "actor": {"login": login}}


def test_ci_activation_and_runner_selection() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert "synchronize" in workflow.get("on", workflow.get(True, {}))["pull_request"]["types"]
    jobs = workflow["jobs"]
    activation = next(step for step in jobs["changes"]["steps"] if step.get("id") == "activation")
    removed_label = [_labeled("maintainer"), {"event": "unlabeled"}]
    paginated = [[{"event": "commented"}], [{"event": "labeled", "actor": None}, _labeled("isaaclab-bot[bot]")]]
    ignored = [[_labeled("github-actions[bot]"), {"event": "commented", "actor": {"login": "human"}}]]
    cases = [
        ("push", "push", "", "", [], True, True, False),
        ("actions bot", "pull_request", "synchronize", "", ignored, False, False, True),
        ("removed human label", "pull_request", "synchronize", "", [removed_label], False, True, True),
        ("paginated bot label", "pull_request", "synchronize", "", paginated, False, True, True),
        ("API error", "pull_request", "synchronize", "", [], True, False, True),
        ("explicit rerun", "pull_request", "labeled", "ci:run-docker", [], True, True, True),
        ("other label", "pull_request", "labeled", "triage", [], True, True, False),
    ]
    data = {
        "script": activation["with"]["script"],
        "continue_on_error": activation.get("continue-on-error", False),
        "cases": cases,
        "runner": jobs["build"]["runs-on"].removeprefix("${{").removesuffix("}}"),
        "curobo": jobs["build-curobo"]["if"],
        "stop": next(step for step in jobs["build"]["steps"] if step.get("name") == "Explain missing PR activation"),
        "cancel": workflow["concurrency"]["cancel-in-progress"].removeprefix("${{").removesuffix("}}"),
    }
    runner = """
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const assert = require('node:assert/strict');
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
(async () => {
  for (const [name, eventName, action, label, pages, error, activated, expectedCancel] of input.cases) {
    const outputs = {};
    const context = {eventName, payload: {action, pull_request: {number: 42}, label: {name: label}},
                     repo: {owner: 'owner', repo: 'lab'}};
    const core = {setOutput: (k,v) => outputs[k] = v};
    const github = {rest: {issues: {listEvents: 'events'}}, paginate: {iterator: async function*() {
      if (error) throw new Error('network unavailable');
      for (const data of pages) yield {data};
    }}};
    try {
      await new AsyncFunction('context', 'github', 'core', input.script)(context, github, core);
    } catch (error) {
      if (!input.continue_on_error) throw new Error(name + ': ' + error.message);
    }
    const needs = {changes: {outputs: {activated: outputs.activated, should_run: 'true'}}};
    const event = {event_name: eventName, event: context.payload};
    const evaluate = code => new Function('needs', 'github', 'fromJSON', `return (${code})`)(needs, event, JSON.parse);
    assert.deepEqual(
      [evaluate(input.runner), evaluate(input.curobo), evaluate(input.stop.if), evaluate(input.cancel)],
      [activated ? ['self-hosted', 'gpu'] : ['ubuntu-latest'],
       activated, !activated, expectedCancel],
      name);
  }
})();
"""
    result = subprocess.run(["node", "-e", runner], input=json.dumps(data), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    stopped = subprocess.run(["bash", "-e", "-c", data["stop"]["run"]], capture_output=True, text=True)
    assert stopped.returncode != 0, "Unactivated PR explanation step exited successfully"


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_benchmark_cancellation(tmp_path: Path, role: str) -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    command = next(
        step["run"]
        for step in workflow["jobs"]["performance-smoke"]["steps"]
        if step.get("name")
        == ("Benchmark the PR base on this runner" if role == "baseline" else "Run runtime benchmarks")
    )
    step_script = tmp_path / "workflow-step.sh"
    step_script.write_text(command, encoding="utf-8")
    controller = tmp_path / "controller"
    scripts = controller / "tools/performance_regression"
    scripts.mkdir(parents=True)
    (controller / "baseline-output").mkdir()
    (scripts / "run_benchmarks.sh").write_bytes(
        (_WORKFLOW.parents[2] / "tools/performance_regression/run_benchmarks.sh").read_bytes()
    )
    (scripts / "source_revision.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "with (Path(os.environ['GITHUB_WORKSPACE']) / 'source-calls').open('a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'prepare':\n"
        "    Path(sys.argv[sys.argv.index('--output') + 1]).parent.mkdir(parents=True, exist_ok=True)\n",
        encoding="utf-8",
    )
    (controller / "benchmark-legs.tsv").write_text("fixture|Fixture-Task|1|600|\n", encoding="utf-8")
    executables = tmp_path / "bin"
    executables.mkdir()
    docker = executables / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json
            import os
            import sys
            import time
            from pathlib import Path

            workspace = Path(os.environ["GITHUB_WORKSPACE"])
            marker = workspace / "running-container"
            if sys.argv[1] == "run" and "--name" in sys.argv:
                name = sys.argv[sys.argv.index("--name") + 1]
                marker.write_text(name, encoding="utf-8")
                print(json.dumps({"pid": os.getpid(), "argv": sys.argv[1:]}), flush=True)
                while marker.exists():
                    time.sleep(0.01)
            elif sys.argv[1] == "rm" and marker.exists():
                assert sys.argv[-1] == marker.read_text(encoding="utf-8")
                marker.unlink()
            """
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    # Exercise Bash cancellation and Docker cleanup; the timeout wrapper is replaced by exec.
    timeout = executables / "timeout"
    timeout.write_text(
        '#!/bin/bash\nprintf "%s" "$1" > "$GITHUB_WORKSPACE/attempt-limit"\nshift\nexec "$@"\n', encoding="utf-8"
    )
    timeout.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(tmp_path),
        "PERF_CONTROLLER": str(controller),
        "PERF_MEASUREMENT_ROOT": str(controller / "measurement"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "CI_IMAGE_TAG": "fixture-image",
        "PERF_BASE_COMMIT": "a" * 40,
        "GITHUB_SHA": "b" * 40,
    }
    expected_mounts = {
        f"{tmp_path}/{path}:/workspace/isaaclab/{path}:{mode}"
        for path, mode in (("source", "rw"), ("scripts", "ro"), ("apps", "ro"))
    }
    for path in MEASUREMENT_PATHS:
        expected_mounts.add(f"{controller}/measurement/{path}:/workspace/isaaclab/{path}:ro")
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        (tmp_path / "source-calls").unlink(missing_ok=True)
        process = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(step_script)],
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            started = next((json.loads(line) for line in process.stdout if line.startswith("{")), None)
            assert started is not None, (stop_signal.name, process.stderr.read())
            docker_args = started["argv"][: started["argv"].index(environment["CI_IMAGE_TAG"])]
            assert (tmp_path / "attempt-limit").read_text() == "600"
            calls = [json.loads(line) for line in (tmp_path / "source-calls").read_text().splitlines()]
            prepared = [call for call in calls if call[0] == "prepare"]
            assert len(prepared) == 1
            args = prepared[0]
            assert args[args.index("--commit") + 1] == ("a" if role == "baseline" else "b") * 40
            assert args[args.index("--measurement-manifest") + 1] == str(
                controller / "measurement/measurement-manifest.json"
            )
            mounts = {docker_args[index + 1] for index, arg in enumerate(docker_args) if arg == "-v"}
            assert expected_mounts <= mounts, (
                f"Missing or incorrect Docker source mounts: {sorted(expected_mounts - mounts)};"
                f" observed: {sorted(mounts)}"
            )
            process.send_signal(stop_signal)
            # GitHub escalates cancellation after 7.5 s + 2.5 s, then kills the process tree.
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 128 + stop_signal, (stop_signal.name, stdout + stderr)
            assert not (tmp_path / "running-container").exists(), stop_signal.name
            try:
                os.kill(started["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f"{stop_signal.name}: benchmark process is still running")
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()


def test_local_pair_ownership_and_workflow_lifecycle() -> None:
    root = _WORKFLOW.parents[2]
    package = root / "tools/performance_regression"
    # Keep paired execution in the workflow and calculations in compare.py. No second controller or embedded tests.
    assert not (package / "pr_comparison.py").exists()
    assert not (_WORKFLOW.parent / "performance-regression-check.yml").exists()
    for path in package.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name.startswith(("test_", "Test"))
            for node in ast.walk(tree)
        ), path
        if path.name == "compare.py":
            assert {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level} <= {
                "contract",
                "metrics",
            }
    assert (root / "tools/perf_smoke_thresholds.json").is_file()
    assert not (root / "tools/performance_regression_thresholds.json").exists()
    # Paired reporting consumes the existing bundle contract; do not add a second metric-selection API.
    assert tuple(signature(extract).parameters) == ("bundle",)
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    job = workflow["jobs"]["performance-smoke"]
    steps = job["steps"]
    names = [step.get("name") for step in steps]
    baseline = names.index("Benchmark the PR base on this runner")
    restore = names.index("Restore the tested PR revision")
    candidate = names.index("Run runtime benchmarks")
    report = names.index("Compare fresh base and PR measurements")
    assert baseline < restore < candidate < report
    assert steps[baseline]["continue-on-error"] is True
    assert workflow["jobs"]["performance-smoke-summary"]["name"] == "performance-smoke"
    conditions = {
        "current": steps[candidate]["if"],
        "restore": steps[restore]["if"],
        "report": steps[report]["if"],
        "budget": job["timeout-minutes"],
    }
    conditions = {name: value.removeprefix("${{").removesuffix("}}") for name, value in conditions.items()}
    evaluated = subprocess.run(
        [
            "node",
            "-e",
            """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const assert = require('assert/strict');
for (const base of ['success', 'failure', 'skipped']) {
    for (const cancelled of [false, true]) {
      const steps = {base_dependencies: {outcome: base}, measure_base: {outcome: base}};
      const github = {event_name: 'pull_request'};
      const evaluate = name => new Function('steps', 'github', 'cancelled', 'always',
        'return (' + input[name] + ')')(steps, github, () => cancelled, () => true);
      assert.equal(evaluate('current'), !cancelled);
      assert.equal(evaluate('restore'), !cancelled);
      assert.equal(evaluate('report'), true);
      assert.equal(evaluate('budget'), 900);
      github.event_name = 'push';
      assert.equal(evaluate('budget'), 450);
    }
}
""",
        ],
        input=json.dumps(conditions),
        capture_output=True,
        text=True,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    # Derive the total timeout from the actual matrix and original three-sample/one-retry policy.
    matrix = next(step["run"] for step in steps if step.get("name") == "Define benchmark matrix")
    limits = [int(line.split("|")[3]) for line in matrix.splitlines() if "|" in line]
    assert sum(limits) * 3 * 2 * 2 / 60 < 900
