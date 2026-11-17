# Performance comparisons

An activated PR update measures its exact base and tested merge result freshly on
the same GPU runner. Open the existing **performance-smoke** check's Actions
Summary to see base FPS, PR FPS, percentage change, and unavailable results for
every configured workload. The existing rolling-history gate is reported separately.

The base is the tested merge commit's first parent, verified against the requested
PR head. Both sides use the PR's saved benchmark package and timer implementation,
with each revision's own dependency image. Source manifests, imported-module checks,
and output hashes bind each sample to the selected checkout and measurement code.
Three independent completed samples, matching workload/hardware/protocol, and valid
source evidence are required before a percentage is shown. FPS is the sample median;
positive change means higher FPS. A zero baseline has no percentage change.

These are descriptive build comparisons, not significance tests or isolated commit
effects. Image and dependency versions can differ. Paired results never set the
rolling-history verdict or change its thresholds. A base setup or measurement failure
leaves the affected comparison unavailable; the current PR is still measured. A
current benchmark failure retains the existing failed-job behavior. Forks can read
the Summary without a write token; history comparisons retain the existing behavior
when no baseline-store credential is available.

The `performance-smoke-<run>-<attempt>` artifact contains both source manifests,
raw samples, source proofs, image IDs, and `paired-comparison.json` / `.md`.
Baseline reuse and a separate fork check publisher are deferred. Keeping both sides
fresh doubles the sample work relative to one side. Per-attempt limits are unchanged:
the four-workload matrix has up to 744 minutes of paired attempts, so PR jobs allow
900 minutes including setup/reporting; other events retain 450 minutes.

## Ownership

Keep execution state in `.github/workflows/build.yaml`: it resolves revisions,
prepares images, preserves the measurement implementation outside checkout cleanup,
and passes explicit paths to the existing launcher. `source_revision.py` owns source
verification and the measurement snapshot; `run_benchmarks.sh` prepares each role's
source manifest once and owns the container lifecycle. The workflow passes the
selected commit and output directory rather than preparing the runner's state.
`cli.py` reads local inputs and writes results; `contract.py` normalizes workload
identity; `compare.py` calculates; `report.py` renders and writes both report shapes.

Use `metrics.extract` with selected names for paired FPS and its default full metric
set for history. Metric paths and numeric validation have one owner in `METRICS` and
its extractor. Use `report.write_json` for both paired and history report JSON. Source
proofs are checked before their exact bytes are decoded as runtime data. The existing
package-install retry helper handles transient dependency-install failures.

Do not add a second paired controller, remote artifact-selection subsystem, network
access to paired calculations/rendering, or tests inside production modules. Keep
history storage in `store.py` and preserve the existing exception, threshold path,
and check names. Extend the existing comparison, source, and workflow tests when
changing these boundaries.

## Validation

Run the CPU tests without starting Isaac Sim:

```bash
uv run --no-project --with-requirements tools/performance_regression/requirements.txt \
  python -m unittest tools.performance_regression.test_compare tools.performance_regression.test_source_revision
uv run --no-project --with pytest --with pyyaml python -m pytest --noconftest \
  .github/actions/_lib/test_performance_workflow.py .github/actions/_lib/test_registry_credential_fallback.py
```
