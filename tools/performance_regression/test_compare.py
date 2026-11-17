# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-only integration tests exercising the real ASV comparison engine."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from . import cli, compare
from .contract import Contract, build
from .metrics import METRICS, PerfSmokeError
from .source_revision import RUNTIME_MODULE, measurement_digest
from .store import BaselineRow


def _measurement(fps: float, startup: float = 10.0) -> dict[str, float]:
    return {"total_fps": fps, "startup_time_s": startup, "gpu_mem_peak_gb": 2.0, "ram_peak_gb": 4.0}


class TestAsvComparison(unittest.TestCase):
    """Check the pinned ASV comparison and the policies retained around it."""

    def setUp(self):
        self.contract = Contract(
            workload={"task": "task", "physics_backend": "physx", "render_backend": None},
            runtime={"gpu_model": "l40s"},
        )
        self.policy = {"defaults": {"warn_regression_pct": 5, "fail_regression_pct": 10}}

    def evaluate(self, baseline, candidate):
        measurements = [_measurement(value) for value in candidate]
        return compare.compare(
            self.contract,
            measurements,
            [_measurement(value) for value in baseline],
            self.policy,
        )

    def test_relative_regressions_and_improvement(self):
        # Three baseline samples exercise the CI fallback; twenty permit ASV's
        # Mann–Whitney test. FPS equality also checks the threshold factor.
        for count in (3, 20):
            for fps, expected in (
                (100, compare.PASS),
                (120, compare.PASS),
                (95, compare.PASS),
                (93, compare.WARN),
                (90, compare.WARN),
                (80, compare.FAIL),
            ):
                with self.subTest(count=count, fps=fps):
                    result = self.evaluate([100] * count, [fps] * 3)
                    self.assertEqual(result.verdict, expected)

        for baseline, candidate, expected in (
            ([60, 100, 140], [50, 80, 110], compare.PASS),
            # An even-sample median must not change when reversing FPS comparisons.
            ([99] * 10 + [101] * 10, [89.995] * 3, compare.FAIL),
            ([100] * 3, [0] * 3, compare.FAIL),
            ([0] * 3, [100] * 3, compare.SKIP),
        ):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertEqual(self.evaluate(baseline, candidate).verdict, expected)

    def test_insufficient_independent_samples_skip(self):
        for baseline, candidate in (([], [80] * 3), ([100] * 2, [80] * 3), ([100] * 3, [80])):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertEqual(self.evaluate(baseline, candidate).verdict, compare.SKIP)

    def test_hard_floor_gates_without_history_even_for_advisory_task(self):
        self.policy.update(
            {
                "per_task_regression_pct": {"task": {"advisory_only": True}},
                "hard_floor_fps": {"l40s": {"task": {"physx": 50}}},
            }
        )
        result = self.evaluate([], [0, 100, 100])
        self.assertEqual(result.verdict, compare.FAIL)
        self.assertIn("hard floor", result.metrics[0].note)
        self.assertIn("hard floor", result.message)

    def test_hard_floor_respects_metric_direction(self):
        thresholds = compare.Thresholds(warn_pct=5, fail_pct=10, hard_floor=15, gating=False)
        breached = compare._evaluate(METRICS[1], [10, 20, 10], [], thresholds, min_samples=3)
        within_limit = compare._evaluate(METRICS[1], [10, 10, 10], [], thresholds, min_samples=3)

        self.assertEqual(breached.verdict, compare.FAIL)
        self.assertTrue(breached.gating)
        self.assertEqual(breached.note, "above hard floor 15")
        self.assertEqual(within_limit.verdict, compare.SKIP)
        self.assertFalse(within_limit.gating)

    def test_advisory_task_does_not_gate(self):
        self.policy["per_task_regression_pct"] = {"task": {"advisory_only": True}}
        result = self.evaluate([100] * 3, [80] * 3)
        self.assertEqual(result.verdict, compare.SKIP)
        self.assertEqual(result.metrics[0].verdict, compare.FAIL)

    def test_higher_is_worse_metrics_are_advisory(self):
        measurements = [_measurement(100, startup=20)] * 3
        result = compare.compare(
            self.contract,
            measurements,
            [_measurement(100)] * 3,
            self.policy,
        )
        self.assertEqual(result.verdict, compare.PASS)
        self.assertEqual(result.metrics[1].verdict, compare.FAIL)

    def test_cli_filters_contracts_and_preserves_failure_in_aggregate(self):
        matching = BaselineRow(self.contract.as_dict(), self.contract.hash, _measurement(100), "commit", "date", "run")
        other = Contract(workload={"task": "other"}, runtime={})
        mismatched = BaselineRow(other.as_dict(), self.contract.hash, _measurement(1), "commit", "date", "run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            bundle.write_text("{}")
            output = root / "combination" / "comparison.json"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(cli.contract_mod, "build", return_value=self.contract),
                patch.object(cli.metrics_mod, "extract", return_value=_measurement(80)),
                patch.object(cli.store_mod, "is_configured", return_value=True),
                patch.object(cli.store_mod, "read", return_value=[matching] * 3 + [mismatched] * 10),
            ):
                status = cli.main(
                    [
                        "compare",
                        "--benchmark_result",
                        str(bundle),
                        str(bundle),
                        str(bundle),
                        "--output_json",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.read_text())["verdict"], compare.FAIL)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["aggregate", "--comparison_dir", directory]), 1)

    def test_cli_infrastructure_errors_and_missing_credentials_do_not_gate(self):
        for configured, expected in ((False, compare.SKIP), (True, compare.ERROR)):
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle.json"
                bundle.write_text("{}")
                output = root / "comparison.json"
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                    patch.object(cli.contract_mod, "build", return_value=self.contract),
                    patch.object(cli.metrics_mod, "extract", return_value=_measurement(80)),
                    patch.object(cli.store_mod, "is_configured", return_value=configured),
                    patch.object(cli.store_mod, "read", side_effect=PerfSmokeError("unavailable")),
                ):
                    status = cli.main(
                        [
                            "compare",
                            "--benchmark_result",
                            str(bundle),
                            "--output_json",
                            str(output),
                        ]
                    )
                self.assertEqual(status, 0)
                self.assertEqual(json.loads(output.read_text())["verdict"], expected)


class TestContracts(unittest.TestCase):
    """Compare solver identity without invalidating legacy single-solver baselines."""

    def test_solver_and_coupling_identity_controls_comparability(self):
        bundle = {
            "run": {
                "task": "task",
                "num_envs": 16,
                "config": {
                    "physics_backend": "newton_mjwarp",
                    "rendering_backend": "none",
                    "presets": [],
                },
            },
            "versions": {"torch": "2.5", "warp": "1.9", "newton": "1.0", "mujoco": "3.4", "mjwarp": "0.1"},
            "hardware": {"gpu_devices": [{"name": "L40S"}], "cpu_name": "cpu"},
        }
        config = bundle["run"]["config"]
        legacy = build(bundle)
        self.assertEqual(legacy.hash, "9138656bb16b2861")
        config["physics_solvers"] = ["newton_mjwarp"]
        config["physics_coupling"] = None
        self.assertEqual(build(bundle), legacy)
        config.update(physics_solvers=["newton_vbd", "newton_mjwarp", "newton_vbd"], physics_coupling="proxy")
        proxy = build(bundle)
        self.assertEqual(proxy.hash, "74606f4967b8b0ba")
        self.assertNotEqual(proxy, legacy)
        config["physics_solvers"] = ["newton_mjwarp", "newton_vbd"]
        self.assertEqual(build(bundle), proxy)
        config["physics_coupling"] = "admm"
        self.assertNotEqual(build(bundle), proxy)
        config["physics_backend"] = "newton_kamino"
        config["physics_solvers"] = ["newton_kamino", "newton_mjwarp"]
        del bundle["versions"]["mjwarp"]
        with self.assertRaisesRegex(compare.PerfSmokeError, "versions.mjwarp"):
            build(bundle)


class TestPairedComparison(unittest.TestCase):
    """Exercise the local CLI, source evidence, calculation, and rendered report together."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.case = 0

    def fixture(self):
        self.case += 1
        self.root = self.directory / str(self.case)
        self.root.mkdir()
        self.legs = self.root / "legs.tsv"
        self.legs.write_text("fixture|Task|16|600|physics=physx\n")
        self.measurement = {"commit": "b" * 40, "files": {"source/isaaclab/isaaclab/benchmark/runtime.py": "c" * 64}}
        self.measurement["digest"] = measurement_digest(self.measurement["files"])
        self.bundles = {}
        self.paths = {}
        for role, root, commit, values in (
            ("baseline", self.root / "baseline", "a" * 40, [90, 100, 1000]),
            ("candidate", self.root, "b" * 40, [80, 81, 82]),
        ):
            root.mkdir(exist_ok=True)
            manifest = {
                "schema_version": 4,
                "commit": commit,
                "files": list(self.measurement["files"]),
                "measurement": self.measurement,
            }
            (root / "source-manifest.json").write_text(json.dumps(manifest))
            (root / "image-id").write_text(f"sha256:{commit}")
            (root / "fixture").mkdir()
            (root / "fixture/status").write_text("ok\n")
            self.bundles[role] = []
            self.paths[role] = []
            for sample, fps in enumerate(values, 1):
                bundle = {
                    "run": {
                        "task": "Task",
                        "status": "completed",
                        "seed": 42,
                        "num_envs": 16,
                        "config": {"physics_backend": "physx", "rendering_backend": "none", "presets": []},
                    },
                    "hardware": {"cpu_name": "CPU", "cpu_count": 32, "gpu_devices": [{"name": "NVIDIA L40S"}]},
                    "versions": {"torch": "2.5", "warp": "1.9", "isaacsim": "6.2"},
                    "runtime": {
                        "total_fps": {"mean": fps},
                        "iterations_completed": 200,
                        "steps_per_iteration": 1,
                        "environment_step_timing": {"warmup_steps": 100, "measurement_mode": "stepping"},
                    },
                }
                path = root / f"fixture/sample-{sample}/benchmark_runtime_fixture.json"
                path.parent.mkdir()
                self.bundles[role].append(bundle)
                self.paths[role].append(path)
        self.save_samples()

    def save_samples(self):
        for role, bundles in self.bundles.items():
            for path, bundle in zip(self.paths[role], bundles):
                data = json.dumps(bundle).encode()
                path.write_bytes(data)
                proof = {
                    "schema_version": 4,
                    "commit": ("a" if role == "baseline" else "b") * 40,
                    "status": "verified",
                    "mismatches": [],
                    "bytecode_policy": "fresh_process_cache",
                    "measurement_digest": self.measurement["digest"],
                    "modules": [{"name": RUNTIME_MODULE, "relative_path": next(iter(self.measurement["files"]))}],
                    "outputs": [{"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
                }
                path.with_name("source-revision.json").write_text(json.dumps(proof))

    def invoke(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(cli.store_mod, "read", side_effect=AssertionError("Paired comparison must stay local")),
            patch.object(cli.store_mod, "write", side_effect=AssertionError("Paired comparison must stay local")),
            patch("urllib.request.urlopen", side_effect=AssertionError("Paired comparison must stay local")),
        ):
            status = cli.main(
                [
                    "pair",
                    "--results_dir",
                    str(self.root),
                    "--legs",
                    str(self.legs),
                    "--base_commit",
                    "a" * 40,
                    "--commit",
                    "b" * 40,
                ]
            )
        self.assertEqual(status, 0, "Informational paired deltas must not change the history gate")
        report = json.loads((self.root / "paired-comparison.json").read_text())
        return report["rows"][0], (self.root / "paired-comparison.md").read_text()

    def test_medians_context_and_non_gating_report(self):
        self.fixture()
        row, markdown = self.invoke()
        self.assertEqual(row["status"], "compared")
        self.assertEqual(row["baseline"]["median"], 100)
        self.assertEqual(row["candidate"]["median"], 81)
        self.assertAlmostEqual(row["change_pct"], -19)
        self.assertIn("-19.00%", markdown)
        self.assertIn("do not change the rolling-history gate", markdown)
        for bundle in self.bundles["candidate"]:
            bundle["versions"]["warp"] = "2.0"
        self.save_samples()
        row, markdown = self.invoke()
        self.assertEqual(row["status"], "compared")
        self.assertTrue(row["context_changed"])
        self.assertIn("Dependency versions differ", markdown)

    def test_pair_reuses_metric_definitions_and_full_extraction_keeps_its_contract(self):
        self.fixture()
        bundle = copy.deepcopy(self.bundles["candidate"][0])
        bundle["runtime"]["startup_time_s"] = {"app_launch": 1, "env_creation": 2, "first_step": 3}
        bundle["resources"] = {"gpu_mem_gb": {"peak": 2}, "ram_gb": {"peak": 4}}
        self.assertEqual(cli.metrics_mod.extract(bundle), _measurement(80, startup=6))
        del bundle["resources"]
        self.assertEqual(cli.metrics_mod.extract(bundle, names=("total_fps",)), {"total_fps": 80})
        with self.assertRaises(PerfSmokeError):
            cli.metrics_mod.extract(bundle)
        with self.assertRaisesRegex(PerfSmokeError, "Unknown metric"):
            cli.metrics_mod.extract(bundle, names=("misspelled",))
        # A definition change must reach the paired path without a second hard-coded FPS path.
        definitions = tuple(
            replace(metric, path=("runtime", "renamed_fps", "mean")) if metric.name == "total_fps" else metric
            for metric in METRICS
        )
        for bundles in self.bundles.values():
            for bundle in bundles:
                bundle["runtime"]["renamed_fps"] = bundle["runtime"].pop("total_fps")
        self.save_samples()
        with patch.object(cli.metrics_mod, "METRICS", definitions):
            row, _ = self.invoke()
        self.assertEqual(row["status"], "compared")
        self.assertAlmostEqual(row["change_pct"], -19)

    def test_incompatible_and_invalid_samples_have_no_delta(self):
        changes = [
            ("run.task", "Other"),
            ("run.num_envs", 32),
            ("run.seed", 43),
            ("run.status", "failed"),
            ("run.config.presets", ["other"]),
            ("hardware.cpu_name", "Other CPU"),
            ("hardware.cpu_name", "unknown"),
            ("hardware.cpu_count", 64),
            ("hardware.gpu_devices", [{"name": "A100"}]),
            ("hardware.gpu_devices", [{"name": "unknown"}]),
            ("runtime.iterations_completed", 201),
            ("runtime.steps_per_iteration", 2),
            ("runtime.environment_step_timing.warmup_steps", 99),
            ("runtime.environment_step_timing.measurement_mode", None),
            ("runtime.total_fps.mean", True),
            ("runtime.total_fps.mean", -1),
            ("runtime.total_fps.mean", float("nan")),
            ("runtime.total_fps.mean", float("inf")),
            ("versions.warp", "2.0"),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.fixture()
                target = self.bundles["candidate"][0]
                parts = field.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                self.save_samples()
                row, markdown = self.invoke()
                self.assertEqual(row["status"], "not_comparable")
                self.assertIsNone(row["change_pct"])
                self.assertIn("Not comparable", markdown)

    def test_missing_corrupt_or_unverified_source_still_reports_the_current_build(self):
        for failure in (
            "missing manifest",
            "missing proof",
            "failed proof",
            "wrong commit",
            "wrong measurement",
            "output tampering",
            "wrong import",
            "bad measurement manifest",
            "invalid proof",
            "non-object JSON",
        ):
            with self.subTest(failure=failure):
                self.fixture()
                path = self.paths["baseline"][0]
                proof_path = path.with_name("source-revision.json")
                proof = json.loads(proof_path.read_text())
                if failure == "missing manifest":
                    (self.root / "baseline/source-manifest.json").unlink()
                elif failure == "missing proof":
                    proof_path.unlink()
                elif failure == "output tampering":
                    path.write_text(path.read_text() + " ")
                elif failure == "non-object JSON":
                    path.write_bytes(b"[]")
                    proof["outputs"][0].update(sha256=hashlib.sha256(b"[]").hexdigest(), bytes=2)
                    proof_path.write_text(json.dumps(proof))
                elif failure == "bad measurement manifest":
                    manifest_path = self.root / "source-manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["measurement"]["digest"] = "d" * 64
                    manifest_path.write_text(json.dumps(manifest))
                else:
                    if failure == "failed proof":
                        proof["status"] = "failed"
                    elif failure == "wrong commit":
                        proof["commit"] = "c" * 40
                    elif failure == "wrong measurement":
                        proof["measurement_digest"] = "d" * 64
                    elif failure == "wrong import":
                        proof["modules"][0]["relative_path"] = "cached-image/runtime.py"
                    elif failure == "invalid proof":
                        proof = {}
                    proof_path.write_text(json.dumps(proof))
                row, markdown = self.invoke()
                self.assertEqual(row["status"], "not_comparable")
                self.assertIsNone(row["change_pct"])
                if failure != "bad measurement manifest":
                    self.assertEqual(row["candidate"]["median"], 81)
                self.assertIn("Not comparable", markdown)

    def test_independent_sample_completeness_and_failed_leg(self):
        for failure in ("missing", "duplicate", "failed leg", "invalid json"):
            with self.subTest(failure=failure):
                self.fixture()
                if failure == "missing":
                    self.paths["baseline"][0].unlink()
                elif failure == "duplicate":
                    self.paths["baseline"][0].rename(
                        self.paths["baseline"][1].with_name("benchmark_runtime_other.json")
                    )
                elif failure == "failed leg":
                    (self.root / "baseline/fixture/status").write_text("failed")
                else:
                    self.paths["baseline"][0].write_text("{")
                row, _ = self.invoke()
                self.assertEqual(row["status"], "not_comparable")
                self.assertIsNone(row["change_pct"])
                self.assertEqual(row["candidate"]["count"], 3)

    def test_zero_baseline_and_escaped_labels(self):
        self.fixture()
        for bundle in self.bundles["baseline"]:
            bundle["runtime"]["total_fps"]["mean"] = 0
        self.save_samples()
        row, markdown = self.invoke()
        self.assertEqual(row["status"], "compared")
        self.assertIsNone(row["change_pct"])
        self.assertIn("base FPS is zero", markdown)
        payload = json.loads((self.root / "paired-comparison.json").read_text())
        payload["rows"][0]["label"] = "task | <script>\nnext"
        rendered = cli.report_mod.render_pair(payload)
        self.assertIn("task &#124; &lt;script&gt; next", rendered)


if __name__ == "__main__":
    unittest.main()
