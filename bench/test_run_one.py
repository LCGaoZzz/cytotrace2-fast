"""Exercise the harness with tiny fake executables, never with real cell data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("run_one.py")
TIME = Path("/usr/bin/time")


@unittest.skipUnless(TIME.is_file(), "GNU /usr/bin/time is required by the harness")
class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.tsv"
        self.input.write_text("tiny fixture\n")
        self.binary = self.root / "fake.py"

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, body):
        self.binary.write_text("#!" + sys.executable + "\n" + body)
        self.binary.chmod(0o755)

    def command(self, label="run"):
        return [sys.executable, str(SCRIPT), "--label", label, "--tier", str(self.input),
                "--anno", str(self.input), "--binary", str(self.binary),
                "--outdir", str(self.root / "runs")]

    def invoke(self, expected, label="run"):
        proc = subprocess.run(self.command(label), text=True, capture_output=True, timeout=12)
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        return proc

    def test_success_and_hashes(self):
        self.fake("import sys,pathlib\np=pathlib.Path(sys.argv[sys.argv.index('-o')+1]); p.mkdir(parents=True)\n(p/'cytotrace2_results.txt').write_text('fixture\\n')\nprint('STAGE_TIMINGS {\"total\":0.1}')\n")
        report = json.loads(self.invoke(0).stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["binary_sha256"]), 64)
        self.assertEqual(len(report["results_sha256"]), 64)
        self.assertIn("peak_rss_gib_time", report)
        self.assertEqual(report["stage_timings"], {"total": 0.1})

    def test_child_failure_propagated(self):
        self.fake("import sys\nsys.exit(7)\n")
        report = json.loads(self.invoke(7).stdout)
        self.assertFalse(report["passed"])
        self.assertEqual(report["rc"], 7)

    def test_missing_result_fails(self):
        self.fake("print('no output')\n")
        self.invoke(1)

    def test_logs_cannot_deadlock_pipe(self):
        self.fake("import sys,pathlib\nprint('x'*1048576)\nprint('y'*1048576,file=sys.stderr)\np=pathlib.Path(sys.argv[sys.argv.index('-o')+1]); p.mkdir(parents=True)\n(p/'cytotrace2_results.txt').write_text('fixture\\n')\n")
        self.invoke(0)

    def test_reused_directory_rejected(self):
        self.fake("print('fixture')\n")
        self.invoke(1)
        self.invoke(2)

    def test_label_cannot_escape_outdir(self):
        self.fake("print('fixture')\n")
        self.invoke(2, "../escape")

    def test_malformed_stage_json_fails(self):
        self.fake("import sys,pathlib\np=pathlib.Path(sys.argv[sys.argv.index('-o')+1]); p.mkdir(parents=True)\n(p/'cytotrace2_results.txt').write_text('fixture\\n')\nprint('STAGE_TIMINGS {bad}')\n")
        self.invoke(1)


if __name__ == "__main__":
    unittest.main()
