import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_two import COLUMNS, compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref, self.new = [Path(self.tmp.name) / name for name in ('ref.tsv', 'new.tsv')]
        self.rows = [['c1', '.20', 'Unipotent', '0', '.21', 'Unipotent'],
                     ['c2', '.40', 'Oligopotent', '.5', '.42', 'Oligopotent'],
                     ['c3', '.60', 'Multipotent', '1', '.61', 'Multipotent']]
        self.write(self.ref, self.rows)
        self.write(self.new, self.rows)

    def write(self, path, rows, header=None):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(['', *COLUMNS] if header is None else header)
            writer.writerows(rows)

    def cli(self, *args):
        run = subprocess.run([sys.executable, '-S', str(Path(__file__).with_name('compare_two.py')),
                              str(self.ref), str(self.new), *args], capture_output=True, text=True)
        report = json.loads(Path(str(self.new) + '.cmp.json').read_text())
        self.assertNotIn('NaN', Path(str(self.new) + '.cmp.json').read_text())
        return run.returncode, report

    def test_identical(self):
        rc, out = self.cli('--byte-check', '--min-spearman', '1')
        self.assertEqual(rc, 0)
        self.assertTrue(out['passed'])
        self.assertEqual(out['spearman_score_pair'], 1)

    def test_header(self):
        self.write(self.new, self.rows, ['', 'wrong', *COLUMNS[1:]])
        self.assertEqual(self.cli()[0], 1)

    def test_header_id_label(self):
        self.write(self.new, self.rows, ['Cell', *COLUMNS])
        self.assertEqual(self.cli()[0], 1)

    def test_wrong_cell(self):
        self.rows[0][0] = 'other'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_cell_order(self):
        self.write(self.new, self.rows[::-1])
        self.assertEqual(self.cli()[0], 1)

    def test_row_count(self):
        self.write(self.new, self.rows[:-1])
        self.assertEqual(self.cli()[0], 1)

    def test_duplicate(self):
        self.rows[0][0] = 'c2'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_empty_id(self):
        self.rows[0][0] = ''
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_wrong_width(self):
        self.rows[0].pop()
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_large_delta(self):
        self.rows[0][1] = '.95'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_potency(self):
        self.rows[0][2] = 'Totipotent'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_one_sided_missing(self):
        for token in ('', 'nan', 'NaN'):
            with self.subTest(token=token):
                self.rows[0][1] = token
                self.write(self.new, self.rows)
                rc, out = self.cli()
                self.assertEqual(rc, 1)
                self.assertEqual(out['columns'][COLUMNS[0]]['mask_mismatch_count'], 1)

    def test_same_missing_mask(self):
        self.rows[0][1] = ''
        self.write(self.ref, self.rows)
        self.rows[0][1] = 'nan'
        self.write(self.new, self.rows)
        rc, out = self.cli()
        self.assertEqual(rc, 0)
        self.assertEqual(out['columns'][COLUMNS[0]]['n_finite'], 2)
        self.assertEqual(self.cli('--byte-check')[0], 1)

    def test_non_finite_and_invalid(self):
        for token in ('inf', '-inf', '1e999', 'bad', '+nan'):
            with self.subTest(token=token):
                self.rows[0][1] = token
                self.write(self.new, self.rows)
                self.assertEqual(self.cli()[0], 1)

    def test_difference_overflow(self):
        self.rows[0][1] = '-1e308'
        self.write(self.ref, self.rows)
        self.rows[0][1] = '1e308'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_tolerance_and_byte_gate(self):
        self.rows[0][1] = '.2000000001'
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 0)
        self.assertEqual(self.cli('--atol', '0')[0], 1)
        self.assertEqual(self.cli('--byte-check')[0], 1)

    def test_no_finite_scores(self):
        for row in self.rows:
            row[1] = ''
        self.write(self.ref, self.rows)
        self.write(self.new, self.rows)
        self.assertEqual(self.cli()[0], 1)

    def test_constant_scores(self):
        for row in self.rows:
            row[1] = '.2'
        self.write(self.ref, self.rows)
        self.write(self.new, self.rows)
        rc, out = self.cli()
        self.assertEqual(rc, 0)
        self.assertIsNone(out['spearman_score_pair'])
        self.assertEqual(self.cli('--min-spearman', '.9')[0], 1)

    def test_invalid_tolerances(self):
        for args in (('--atol', '-1'), ('--rtol', 'nan'), ('--min-spearman', 'nan')):
            self.assertEqual(self.cli(*args)[0], 1)

    def test_missing_file(self):
        self.new.unlink()
        self.assertEqual(self.cli()[0], 1)

    def test_empty_input(self):
        self.write(self.new, [])
        self.assertEqual(self.cli()[0], 1)

    def test_spearman_ties(self):
        self.rows[0][1] = '.4'
        self.write(self.ref, self.rows)
        self.write(self.new, self.rows)
        self.assertEqual(compare(self.ref, self.new)['spearman_score_pair'], 1)


if __name__ == '__main__':
    unittest.main()
