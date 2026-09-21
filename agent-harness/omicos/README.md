# Omicos integration

Agent ID: `cytotrace2_fast_analyst`. Skill ID: `cytotrace2-fast`.
The bundle mirrors iobrx's Agent/Skill frontmatter and portable resource layout.
It reuses the host's shell, Python interpreter, session and job tools; no new
service, MCP dependency or Omicos-core change is needed.

## Install into a development workspace

From the cytotrace2-fast checkout:

```bash
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace
```

This creates:

```text
agents/cytotrace2_fast_analyst.md
skills/cytotrace2-fast/SKILL.md
skills/cytotrace2-fast/scripts/run_cytotrace2.py
skills/cytotrace2-fast/scripts/prepare_h5ad.py
skills/cytotrace2-fast/references/...
```

Both destinations are preflighted. Existing installations (including dangling
symlinks) are refused, and a failed copy removes only files/directories created
by that invocation. Empty parent directories may remain. No force/overwrite
mode is provided. The copied Skill does not import from this checkout.

Use your Omicos deployment's authorized workspace extension discovery. Following
the reference iobrx development configuration, a maintainer can configure:

```bash
export OMICOS_TEMPLATES_DIR=/path/to/analysis-workspace
export OMICOS_SKILL_ROOTS=/path/to/analysis-workspace/skills
export OMICOS_AGENTS_OFFLINE=1
export OMICOS_SKILLS_OFFLINE=1
```

Discovery remains subject to host capabilities and permissions; copying files is
not proof that a live account can use them. Verify Agent/Skill listing and resolve
`scripts/run_cytotrace2.py` with `skill_resource`, `include_runtime_path: true`.
Run its `doctor` command using the existing analysis interpreter and a prepared
request. Live Omicos discovery was not executed by the adapter unit tests.

## Prepare a shared catalog contribution

```bash
python agent-harness/install_omicos.py \
  --destination /path/to/omicos-admin-checkout --layout catalog
```

The destination is `domains/biology/agents/cytotrace2_fast_analyst.md` and
`domains/biology/skills/cytotrace2-fast/`, not legacy top-level catalog folders.
Use a separate admin feature branch, run that checkout's current validators and
submit a separate PR. Its review/deployment is distinct from this repository PR.
The catalog layout is based on the linked iobrx reference; verify against the
actual admin checkout rather than assuming a live catalog deployment succeeded.

## Runtime

Prepare the Rust binary and original assets once; see the repository's build
instructions. Supply explicit asset and binary paths in each saved request.
Use host-native Linux/macOS/WSL paths. The Rust implementation uses Unix file I/O;
do not point Windows Python at a Linux executable without running inside WSL.
No installer builds the binary, downloads packages or changes global settings.
See [runtime and interpretation](skills/cytotrace2-fast/references/runtime-and-interpretation.md).
