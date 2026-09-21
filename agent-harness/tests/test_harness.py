"""Contract tests. Fake-engine tests do NOT establish CytoTRACE2 numerical parity."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "omicos" / "skills" / "cytotrace2-fast" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))
import run_cytotrace2 as runner
import prepare_h5ad as prepare
import install_omicos as installer

FAKE = r'''
import os, pathlib, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("cytotrace2-fast 1.3.1")
    sys.exit(0)
values = dict(zip(args[:-1:2], args[1:-1:2]))
assert args[-1] == "--disable-plotting"
assert set(values) == {"--input-path", "--output-dir", "--assets", "--species", "--seed",
                       "--batch-size", "--smooth-batch-size", "--max-cores"}
root = pathlib.Path(__file__).parent
(root / "worker.pid").write_text(str(os.getpid()))
mode = (root / "mode").read_text() if (root / "mode").exists() else "ok"
if mode == "timeout":
    time.sleep(60)
if mode == "fail":
    print("deliberate fake failure", file=sys.stderr)
    sys.exit(7)
cells = pathlib.Path(values["--input-path"]).read_text().splitlines()[0].split("\t")
if mode == "mismatch":
    cells.reverse()
out = pathlib.Path(values["--output-dir"])
header = "\tCytoTRACE2_Score\tCytoTRACE2_Potency\tCytoTRACE2_Relative\tpreKNN_CytoTRACE2_Score\tpreKNN_CytoTRACE2_Potency\n"
if mode == "schema":
    header = header.replace("CytoTRACE2_Relative", "Wrong")
with (out / "cytotrace2_results.txt").open("w") as f:
    f.write(header)
    for i, cell in enumerate(cells):
        score = "nan" if mode == "allnan" or (mode == "nan" and i == 0) else "0.2"
        f.write(f"{cell}\t{score}\tUnipotent\t0.5\t0.2\tUnipotent\n")
print("fake engine; NOT biological inference")
'''


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "counts.tsv"
        self.input.write_text("cell-A\tcell-B\nGeneA\t1\t4\nGeneB\t3\t0\n")
        self.assets = self.root / "assets"
        self.assets.mkdir()
        names = ["MANIFEST.json", "features.txt", "background.npz", "map_human_alias.tsv",
                 "map_human_ortholog.tsv", "map_mouse_alias.tsv"] + [f"model{i}.npz" for i in range(1, 20)]
        for name in names:
            (self.assets / name).write_text("fixture, not a real model")
        self.binary = self.root / "fake cytotrace2"
        self.binary.write_text("#!" + sys.executable + "\n" + FAKE)
        self.binary.chmod(0o755)
        self.request_path = self.root / "request.json"
        self.raw = {"input_path": "counts.tsv", "output_dir": "run", "species": "human",
                    "expression_scale": "counts", "assets": "assets", "binary": str(self.binary)}
        self.request_path.write_text(json.dumps(self.raw))

    def request(self, **changes):
        self.request_path.write_text(json.dumps({**self.raw, **changes}))
        return runner.load_request(self.request_path)

    def mode(self, mode):
        (self.root / "mode").write_text(mode)

    def test_relative_paths_and_defaults(self):
        request = self.request()
        self.assertEqual(request["input_path"], str(self.input))
        self.assertEqual((request["max_cores"], request["seed"]), (4, 14))

    def test_invalid_requests(self):
        for change in ({"seed": -1}, {"seed": 2**32}, {"seed": True}, {"max_cores": 0},
                       {"batch_size": 0}, {"species": "rat"}, {"expression_scale": "log1p"},
                       {"timeout_seconds": float("nan")}, {"timeout_seconds": True}, {"unknown": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.request(**change)

    def test_input_native_header_and_full_validation(self):
        cells, info = runner.inspect_input(self.input, full=True)
        self.assertEqual(cells, ["cell-A", "cell-B"])
        self.assertEqual(info["genes"], 2)
        self.assertFalse(info["expression_scale_verified"])

    def test_invalid_inputs(self):
        for text in ("gene\tA\tB\nG\t1\t2\n", "\tA\tB\nG\t1\t2\n",
                     "A\tA\nG\t1\t2\n", "A\tB\nG\t1\t2\nG\t3\t4\n",
                     "A\tB\nG\t-1\t2\n", "A\tB\nG\tnan\t2\n", "A\tB\nG\t0\t0\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.input.write_text(text)
                runner.inspect_input(self.input, full=True)

    def test_success_and_nonoverwrite(self):
        request = self.request(max_cores=2, seed=123)
        original = self.input.read_bytes()
        result = runner.execute(request, full=True)
        self.assertEqual(result["status"], "completed")
        self.assertIn("--max-cores", result["command"])
        self.assertEqual(result["engine_environment"]["RAYON_NUM_THREADS"], "2")
        self.assertEqual(original, self.input.read_bytes())
        self.assertTrue((self.root / "run" / "summary.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.execute(request)

    def test_nonzero_exit_is_recorded(self):
        self.mode("fail")
        with self.assertRaises(ValueError):
            runner.execute(self.request())
        record = json.loads((self.root / "run" / "results_manifest.json").read_text())
        self.assertEqual((record["status"], record["returncode"]), ("failed", 7))
        self.assertNotIn("results", record["artifacts"])
        self.assertIn("deliberate", (self.root / "run" / "stderr.log").read_text())

    def test_timeout_is_recorded(self):
        self.mode("timeout")
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.execute(self.request(timeout_seconds=0.1))
        record = json.loads((self.root / "run" / "results_manifest.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("TimeoutExpired", record["error"])

    def test_result_missingness_is_visible(self):
        self.mode("nan")
        result = runner.execute(self.request())
        self.assertEqual(result["status"], "completed_with_warnings")
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["missing_numeric"]["CytoTRACE2_Score"], 1)

    def test_bad_results_are_rejected(self):
        for mode in ("mismatch", "schema", "allnan"):
            self.mode(mode)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.execute(self.request(output_dir=mode))

    def test_sigterm_reaps_worker_and_records_interruption(self):
        self.mode("timeout")
        proc = subprocess.Popen([sys.executable, str(SCRIPTS / "run_cytotrace2.py"),
                                 "run", "--request", str(self.request_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            while not (self.root / "worker.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue((self.root / "worker.pid").exists())
            pid = int((self.root / "worker.pid").read_text())
            proc.terminate()
            proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 130)
            record = json.loads((self.root / "run" / "results_manifest.json").read_text())
            self.assertEqual(record["status"], "interrupted")
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()

    def test_portable_frontmatter(self):
        agent = (ROOT / "omicos" / "agents" / installer.AGENT).read_text()
        skill = (SCRIPTS.parent / "SKILL.md").read_text()
        self.assertTrue(agent.startswith("---\n"))
        self.assertIn("id: cytotrace2_fast_analyst\n", agent)
        self.assertIn("skills:\n  - cytotrace2-fast\n", agent)
        self.assertIn("execution_mode: packaged_python\n", skill)
        self.assertIn("runtime_entrypoint: scripts/run_cytotrace2.py\n", skill)

    def test_old_or_wrong_binary_rejected(self):
        for version in ("cytotrace2-fast 1.3.0", "cytotrace2-py 1.1.0"):
            self.binary.write_text("#!" + sys.executable + "\nprint(" + repr(version) + ")\n")
            with self.subTest(version=version), self.assertRaises(ValueError):
                runner.runtime(self.request())

    def test_missing_asset_rejected(self):
        (self.assets / "model19.npz").unlink()
        with self.assertRaisesRegex(ValueError, "asset"):
            runner.runtime(self.request())

    def test_copied_runtime_is_independent(self):
        workspace = self.root / "workspace"
        installed = installer.install(workspace)
        entry = Path(installed[1]) / "scripts" / "run_cytotrace2.py"
        run = subprocess.run([sys.executable, str(entry), "run", "--request", str(self.request_path)],
                             cwd=self.root, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["status"], "completed")
        status = subprocess.run([sys.executable, str(entry), "status", "--output-dir", str(self.root / "run")],
                                text=True, capture_output=True)
        self.assertEqual(json.loads(status.stdout)["status"], "completed")

    def test_installer_layout_preflight_and_no_pyc(self):
        workspace = self.root / "workspace"
        skill = workspace / "skills" / "cytotrace2-fast"
        skill.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            installer.install(workspace)
        self.assertFalse((workspace / "agents" / installer.AGENT).exists())
        catalog = self.root / "admin"
        paths = installer.install(catalog, "catalog")
        self.assertTrue(all("domains/biology/" in p for p in paths))
        self.assertFalse(list(catalog.rglob("*.pyc")))
        with self.assertRaises(FileExistsError):
            installer.install(catalog, "catalog")

    def test_installer_rolls_back_partial_copy(self):
        root = self.root / "rollback"
        with patch.object(installer.shutil, "copytree", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                installer.install(root)
        self.assertFalse((root / "agents" / installer.AGENT).exists())
        self.assertFalse((root / "skills" / installer.SKILL).exists())


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"), "optional matrix dependencies")
class ExportTests(unittest.TestCase):
    def test_dense_and_sparse_export(self):
        import numpy as np
        from scipy.sparse import csr_matrix, csc_matrix
        matrix = np.array([[1., 0.], [2., 3.]])
        for x in (matrix, csr_matrix(matrix), csc_matrix(matrix)):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "counts.tsv"
                prepare.export_matrix(x, ["G1", "G2"], ["C1", "C2"], path, {}, 1)
                self.assertEqual(path.read_text(), "C1\tC2\nG1\t1\t2\nG2\t0\t3\n")
                self.assertEqual(runner.inspect_input(path, True)[1]["genes"], 2)
                with self.assertRaises(ValueError):
                    prepare.export_matrix(x, ["G1", "G2"], ["C1", "C2"], path, {})

    def test_export_failure_removes_partial_outputs(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counts.tsv"
            with self.assertRaises(ValueError):
                prepare.export_matrix(np.array([[1., -1.]]), ["G1", "G2"], ["C1"], path, {}, 1)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".provenance.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("anndata"), "optional anndata not installed")
    def test_real_h5ad_slot_selection(self):
        import anndata as ad
        import numpy as np
        from scipy.sparse import csr_matrix
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = ad.AnnData(np.array([[11., 12.], [13., 14.]]))
            data.obs_names, data.var_names = ["C1", "C2"], ["G1", "G2"]
            data.layers["counts"] = csr_matrix([[1., 0.], [2., 3.]])
            data.raw = data.copy()
            data = data[:, [0]].copy()  # raw still has both genes.
            data.write_h5ad(root / "input.h5ad")
            for slot, genes in (("X", 1), ("layers/counts", 1), ("raw.X", 2)):
                out = root / (slot.replace("/", "_") + ".tsv")
                result = subprocess.run([sys.executable, str(SCRIPTS / "prepare_h5ad.py"),
                                         "--input", str(root / "input.h5ad"), "--output", str(out),
                                         "--matrix-source", slot, "--expression-scale", "counts"],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(runner.inspect_input(out, True)[1]["genes"], genes)
                if slot == "layers/counts":
                    self.assertIn("G1\t1\t2", out.read_text())


if __name__ == "__main__":
    unittest.main()
