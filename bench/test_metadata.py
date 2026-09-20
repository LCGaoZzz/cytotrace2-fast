"""Fast release metadata checks; no model execution or external dependencies."""
import csv
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_single_source_version(self):
        cargo = (ROOT / 'Cargo.toml').read_text()
        version = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
        lock = (ROOT / 'Cargo.lock').read_text()
        own = re.search(r'name = "cytotrace2-fast"\nversion = "([^"]+)"', lock).group(1)
        self.assertEqual(version, own)
        self.assertIn('const VERSION: &str = env!("CARGO_PKG_VERSION");',
                      (ROOT / 'src/main.rs').read_text())

    def test_legacy_benchmark_units(self):
        with (ROOT / 'bench/benchmark_results.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertNotIn(None, row)
            for key in ('official_wall_s', 'fast_wall_s', 'speedup',
                        'official_peak_rss_gb', 'fast_peak_rss_gb'):
                if row[key]:
                    self.assertTrue(math.isfinite(float(row[key])))
                    self.assertGreater(float(row[key]), 0)
            if row['speedup']:
                ratio = float(row['official_wall_s']) / float(row['fast_wall_s'])
                self.assertAlmostEqual(ratio, float(row['speedup']), delta=.01)

    def test_manycore_provenance_and_ratios(self):
        with (ROOT / 'bench/manycore_results.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertNotIn(None, row)
            self.assertEqual(row['reference_version'], 'cytotrace2-fast-1.2.0')
            self.assertEqual(row['candidate_source_snapshot'],
                             'aebce88d422c9bc7ea16d185e1e30508aa446cf3')
            single = row['dataset'] in ('synthetic_100k', 'synthetic_250k')
            self.assertEqual(row['candidate_runs'], '1' if single else '3')
            self.assertEqual(row['candidate_statistic'], 'single_diagnostic' if single else 'median')
            self.assertIn('NOT finalized HEAD', row['notes'])
            for ref, speedup in (('reference_default_wall_s', 'speedup_vs_default'),
                                 ('reference_groups224_wall_s', 'speedup_vs_groups224')):
                self.assertEqual(bool(row[ref]), bool(row[speedup]))
                if row[ref]:
                    self.assertAlmostEqual(float(row[ref]) / float(row['candidate_wall_s']),
                                           float(row[speedup]), places=5)
