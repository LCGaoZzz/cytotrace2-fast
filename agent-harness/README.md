# cytotrace2-fast agent harness

Portable **Omicos Agent + Skill** for the existing Rust executable. This is an
integration bundle, not another implementation of CytoTRACE2 and not a new MCP
server. No model assets, binaries, credentials or machine-specific paths are bundled.

Start with [Omicos installation](omicos/README.md), then the
[Skill](omicos/skills/cytotrace2-fast/SKILL.md). The layout follows
[iobrx's Omicos harness](https://github.com/LCGaoZzz/iobrx/tree/5ca23a985d722ce346e79846a90e6e1b9161e179/agent-harness/omicos).

The Python runner uses only the standard library (Python >=3.10), invokes
`cytotrace2-fast >=1.3.1`, and records commands, versions, logs, results and
missing-value diagnostics. The optional h5ad exporter needs anndata/numpy/scipy.
Both scripts travel inside the Skill's `scripts/` directory.

```bash
python agent-harness/install_omicos.py --destination /path/to/workspace
python -m unittest discover -s agent-harness/tests -v
```

The request template is [examples/request.json](examples/request.json); replace
its input path and set the actual binary/assets paths. This PR does not change
Rust algorithms, assets, Cargo dependencies, or benchmark claims, and does not
install anything into a production Omicos catalog.

## Validation scope

Tests use a deliberately fake executable to exercise subprocess arguments,
version rejection, nonzero exits, timeout cleanup, result schema/ID checks,
missingness, portable installation and non-overwrite behavior. Dense and sparse
export tests exercise the actual exporter. A separate optional test exercises
real h5ad files, including `raw.X` with a different gene dimension; it skips when
anndata is unavailable. CI installs the optional dependencies to run that test.
These are adapter tests, **not model inference, biological validation or numerical
parity against official CytoTRACE2**. Existing Rust/parity tests remain separate.
