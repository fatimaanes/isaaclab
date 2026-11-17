# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the selected Git checkout and the benchmark's imported source paths."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path, PurePosixPath

RUNTIME_MODULE = "isaaclab.benchmark.entrypoints.runtime"
MEASUREMENT_PATHS = ("source/isaaclab/isaaclab/benchmark", "source/isaaclab/isaaclab/utils/timer.py")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Git source inspection failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def measurement_digest(files: dict[str, str]) -> str:
    """Identify measurement contents independently of the commit containing them."""
    return _digest(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def resolve_base(root: Path, expected_commit: str, expected_head: str) -> str:
    """Return the tested merge's first parent, rather than a possibly stale event base."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    parents = [
        line.split()[1]
        for line in _git(root, "cat-file", "-p", commit).decode().split("\n\n", 1)[0].splitlines()
        if line.startswith("parent ")
    ]
    if commit != expected_commit or len(parents) != 2 or parents[1] != expected_head:
        raise ValueError("The checkout is not the expected PR merge with the requested head as its second parent.")
    return parents[0]


def snapshot_measurement(root: Path, output_dir: Path) -> dict:
    """Save the tested revision's measurement package for both benchmark legs."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    entries = {}
    for entry in _git(root, "ls-tree", "-rz", "--full-tree", commit, "--", *MEASUREMENT_PATHS).split(b"\0")[:-1]:
        metadata, path_bytes = entry.split(b"\t", 1)
        _, kind, oid = metadata.decode().split()
        if kind == "blob":
            entries[path_bytes.decode()] = oid
    if not entries:
        raise ValueError("The selected PR contains no measurement implementation.")
    batch = io.BytesIO(_git(root, "cat-file", "--batch", data="".join(f"{oid}\n" for oid in entries.values()).encode()))
    files = {}
    for path in entries:
        _, _, size = batch.readline().split()
        blob = batch.read(int(size))
        batch.read(1)
        destination = output_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        files[path] = _digest(blob)
    manifest = {"schema_version": 1, "commit": commit, "files": files, "digest": measurement_digest(files)}
    (output_dir / "measurement-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def check_checkout(root: Path, expected_commit: str | None = None) -> str:
    """Check the selected revision and tracked Python source on the host."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"Checked-out revision {commit} differs from expected commit {expected_commit}.")
    changed = _git(
        root,
        "-c",
        "diff.autoRefreshIndex=true",
        "-c",
        "core.filemode=false",
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        commit,
        "--",
        "source",
        "scripts",
    )
    for path in changed.decode().split("\0"):
        if path.endswith((".py", ".pyi")):
            details = _git(root, "diff", "--numstat", "--summary", "--no-renames", commit, "--", path).decode().strip()
            raise ValueError(f"Checked-out Python source differs from commit {commit}: {path}\n{details}")
    return commit


def prepare_manifest(
    root: Path, expected_commit: str | None = None, *, measurement_manifest: dict | None = None
) -> dict:
    """Record the selected commit and repository-relative Python paths.
    `files` lists paths. `namespaces` maps installed package names to their source directories.
    """
    commit = check_checkout(root, expected_commit)
    files = sorted(
        path
        for path in _git(root, "ls-tree", "-rz", "--name-only", commit, "--", "source", "scripts").decode().split("\0")
        if path.endswith((".py", ".pyi"))
    )
    if not files:
        raise ValueError("The selected commit contains no Python source under source/ or scripts/.")
    namespaces = {}
    for path in files:
        parts = PurePosixPath(path).parts
        # Test directories may contain __init__.py without being installed packages.
        if len(parts) >= 4 and parts[0] == "source" and parts[2] == parts[1].replace("-", "_"):
            prefix = "/".join(parts[:3])
            namespaces.setdefault(parts[2], set()).add(prefix)
    manifest = {
        "schema_version": 4,
        "commit": commit,
        "files": files,
        "namespaces": {name: sorted(paths) for name, paths in sorted(namespaces.items())},
    }
    if measurement_manifest is not None:
        try:
            if not isinstance(measurement_manifest["files"], dict):
                raise TypeError("Measurement files must map paths to hashes.")
            if measurement_digest(measurement_manifest["files"]) != measurement_manifest["digest"]:
                raise ValueError("Measurement manifest digest does not match its files.")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid measurement manifest: {exc}") from exc
        manifest["measurement"] = measurement_manifest
    return manifest


def read_verified_result(manifest: dict, proof: dict, path: Path, expected_commit: str, measurement: dict) -> dict:
    """Read runtime data only after its source evidence and exact output bytes are verified."""
    if manifest.get("schema_version") != 4 or not expected_commit or manifest.get("commit") != expected_commit:
        raise ValueError("Source manifest does not identify the intended checkout.")
    if not measurement.get("files") or measurement_digest(measurement["files"]) != measurement.get("digest"):
        raise ValueError("Shared measurement manifest is missing or invalid.")
    if manifest.get("measurement") != measurement:
        raise ValueError("The sample does not use the shared PR measurement implementation.")
    if (
        proof.get("schema_version") != 4
        or proof.get("status") != "verified"
        or proof.get("mismatches") != []
        or proof.get("commit") != expected_commit
        or proof.get("bytecode_policy") != "fresh_process_cache"
        or proof.get("measurement_digest") != measurement["digest"]
    ):
        raise ValueError("Source proof is missing, failed, or identifies another checkout/measurement.")
    modules = proof.get("modules", [])
    allowed = set(manifest["files"]) | measurement["files"].keys()
    if not any(module.get("name") == RUNTIME_MODULE for module in modules) or any(
        module.get("relative_path") not in allowed for module in modules
    ):
        raise ValueError("Imported benchmark source does not match the source manifest.")
    outputs = [item for item in proof.get("outputs", []) if item.get("path") == path.name]
    data = path.read_bytes()
    if len(outputs) != 1 or outputs[0].get("sha256") != _digest(data) or outputs[0].get("bytes") != len(data):
        raise ValueError("Runtime result bytes do not match their verified source proof.")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Runtime sample must be a JSON object.")
    return result


def _inspect_files(files: dict, root: Path) -> list[dict]:
    mismatches = []
    for relative, expected in files.items():
        try:
            if _digest((root / relative).read_bytes()) != expected:
                mismatches.append(
                    {"path": relative, "reason": "Measurement source differs from the selected manifest."}
                )
        except OSError as exc:
            mismatches.append({"path": relative, "reason": f"Measurement source could not be read: {exc.strerror}."})
    return mismatches


def _inspect_modules(manifest: dict, root: Path) -> tuple[list[dict], list[dict]]:
    records, mismatches = [], []
    measurement_files = manifest.get("measurement", {}).get("files", {})
    files = set(manifest["files"]) | measurement_files.keys()
    for name, module in sorted(list(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        if not filename:
            # Modules without a source file have no path to verify.
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if name.partition(".")[0] not in manifest["namespaces"] and relative not in files:
            continue
        record = {
            "name": name,
            "path": str(path),
            "relative_path": relative,
        }
        if relative not in files:
            mismatches.append(
                {
                    "module": name,
                    "path": str(path),
                    "reason": "Imported Python source is absent from the selected commit manifest.",
                }
            )
        records.append(record)
    return records, mismatches


def _output_state(output_dir: Path) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size) for path in output_dir.rglob("benchmark_runtime_*.json")
    }


def run_benchmark(manifest: dict, root: Path, output_dir: Path, argv: list[str]) -> int:
    """Verify source evidence after the CLI completes successfully."""
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _output_state(output_dir)
    with tempfile.TemporaryDirectory(prefix="isaaclab-source-pycache-") as pycache:
        sys.pycache_prefix = pycache
        try:
            sys.argv = ["isaaclab", *argv]
            from isaaclab.cli import cli

            cli()
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        modules, mismatches = _inspect_modules(manifest, root)
        if RUNTIME_MODULE not in sys.modules:
            mismatches.append(
                {"module": RUNTIME_MODULE, "reason": "Runtime entrypoint was not imported in this process."}
            )
        measurement = manifest.get("measurement")
        if measurement is not None:
            if measurement_digest(measurement["files"]) != measurement["digest"]:
                mismatches.append({"reason": "Measurement manifest digest does not match its files."})
            mismatches.extend(_inspect_files(measurement["files"], root))
        outputs = []
        for path, state in sorted(_output_state(output_dir).items()):
            if before.get(path) == state:
                continue
            data = path.read_bytes()
            outputs.append(
                {"path": path.relative_to(output_dir).as_posix(), "sha256": _digest(data), "bytes": len(data)}
            )
        if not outputs:
            mismatches.append({"reason": "The benchmark invocation produced no runtime JSON output."})
        verified = not mismatches
        sidecar = {
            "schema_version": 4,
            "status": "verified" if verified else "failed",
            "commit": manifest["commit"],
            "bytecode_policy": "fresh_process_cache",
            "modules": modules,
            "outputs": outputs,
            "mismatches": mismatches,
        }
        if measurement is not None:
            sidecar["measurement_digest"] = measurement["digest"]
        (output_dir / "source-revision.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        print(f"Source revision {manifest['commit']}: {sidecar['status']} ({len(modules)} imported modules checked)")
        for mismatch in mismatches:
            context = mismatch.get("module", mismatch.get("path", "runtime output"))
            print(f"Source verification: {context}: {mismatch['reason']}", file=sys.stderr)
    return 0 if verified else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkout-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--commit")
    prepare.add_argument("--measurement-manifest", type=Path)
    resolve = commands.add_parser("resolve-base")
    resolve.add_argument("--checkout-root", type=Path, required=True)
    resolve.add_argument("--commit", required=True)
    resolve.add_argument("--head-commit", required=True)
    check = commands.add_parser("check")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--checkout-root", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--checkout-root", type=Path, required=True)
    snapshot.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--checkout-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("benchmark_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve-base":
            print(resolve_base(args.checkout_root, args.commit, args.head_commit))
            return 0
        if args.command == "snapshot":
            manifest = snapshot_measurement(args.checkout_root, args.output_dir)
            print(f"Prepared measurement source {manifest['commit']} ({manifest['digest']})")
            return 0
        if args.command == "prepare":
            measurement = json.loads(args.measurement_manifest.read_text()) if args.measurement_manifest else None
            manifest = prepare_manifest(args.checkout_root, args.commit, measurement_manifest=measurement)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"Prepared source revision {manifest['commit']} ({len(manifest['files'])} Python files)")
            return 0
        if args.command == "check":
            try:
                manifest = json.loads(args.manifest.read_text())
                check_checkout(args.checkout_root, manifest["commit"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                proof_path = args.output_dir / "source-revision.json"
                if proof_path.is_file():
                    proof = json.loads(proof_path.read_text())
                    proof["status"] = "failed"
                    proof["mismatches"].append({"reason": f"Host Git checkout check failed: {exc}"})
                    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
                raise
            return 0
        benchmark_argv = args.benchmark_argv
        if benchmark_argv[:1] == ["--"]:
            benchmark_argv = benchmark_argv[1:]
        return run_benchmark(json.loads(args.manifest.read_text()), args.checkout_root, args.output_dir, benchmark_argv)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        traceback.print_exc()
        print(f"Source revision verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
