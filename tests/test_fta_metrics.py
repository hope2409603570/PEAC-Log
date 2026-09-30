import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

STAGE = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, STAGE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

legacy = load('fta_legacy', 'utils/evaluate.py')
b0 = load('fta_b0', 'experiments/b0_evaluator.py')

class FtaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, labels, ids=None):
        path = self.root / name
        ids = ids or list(range(1, len(labels) + 1))
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['LineId', 'EventTemplate'])
            writer.writerows(zip(ids, labels))
        return path

    def check_both(self, gt, pred, expected):
        gt_path = self.write('gt.csv', gt)
        pred_path = self.write('pred.csv', pred)
        output = self.root / 'summary.csv'
        scores = legacy.evaluate_metrics(gt_path, pred_path, 'fixture', output_csv_path=output)
        for score, value in zip(scores, expected):
            self.assertAlmostEqual(score, value)
        normalized_gt = [b0.normalize_template(label) for label in gt]
        normalized_pred = [b0.normalize_template(label) for label in pred]
        for metrics in (b0.evaluate_prediction(gt_path, pred_path),
                        b0.evaluate_aligned(map(str, range(1, len(gt)+1)), normalized_gt, normalized_pred)):
            for key, value in zip(('PA', 'GA', 'FGA', 'FTA'), expected):
                self.assertAlmostEqual(metrics[key], value)
            self.assertEqual(metrics['FTA_definition'], legacy.FTA_DEFINITION)
        return gt_path, pred_path, output

    def test_perfect_and_normalization(self):
        self.check_both(['A <NUM>', 'B: <*>'], ['a <IP>', 'b <*>'], (1, 1, 1, 1))

    def test_correct_template_on_incomplete_group_can_exceed_fga(self):
        # A is fully recovered; B is correctly named but incomplete; C is polluted.
        self.check_both(['A <*>', 'A <*>', 'B <*>', 'B <*>', 'C'],
                        ['A <*>', 'A <*>', 'B <*>', 'C', 'C'], (.8, .4, 1/3, 2/3))

    def test_exact_groups_with_wrong_template_names(self):
        self.check_both(['A', 'A', 'B'], ['X', 'X', 'Y'], (0, 1, 1, 0))

    def test_merge_and_split(self):
        self.check_both(['A', 'A', 'B', 'B'], ['A', 'A', 'A', 'A'], (.5, 0, 0, 0))
        self.check_both(['A', 'A', 'B', 'B'], ['A', 'X', 'B', 'B'], (.75, .5, .4, .8))

    def test_shuffled_and_non_numeric_ids(self):
        gt = self.write('gt.csv', ['A', 'A', 'B'], ['id1', 'id2', 'id3'])
        pred = self.write('pred.csv', ['B', 'A', 'A'], ['id3', 'id1', 'id2'])
        scores = legacy.evaluate_metrics(gt, pred, output_csv_path=self.root/'summary.csv')
        self.assertEqual(scores, (1, 1, 1, 1))
        metrics = b0.evaluate_prediction(gt, pred)
        self.assertEqual(metrics['FTA'], 1)
        self.assertFalse(metrics['alignment']['order_equal'])

    def test_missing_extra_duplicate_and_empty_inputs_fail(self):
        gt = self.write('gt.csv', ['A', 'B'])
        for labels, ids in ((['A'], [1]), (['A', 'B', 'C'], [1,2,3]), (['A', 'A'], [1,1]), (['A', ''], [1,2]), ([], [])):
            pred = self.write('pred.csv', labels, ids)
            with self.assertRaises(ValueError):
                legacy.evaluate_metrics(gt, pred, output_csv_path=self.root/'summary.csv')
            with self.assertRaises(b0.EvaluationError):
                b0.evaluate_prediction(gt, pred)

    def test_both_files_with_duplicate_ids_fail(self):
        gt = self.write('gt.csv', ['A', 'B'], [1,1])
        pred = self.write('pred.csv', ['A', 'B'], [1,1])
        with self.assertRaises(b0.EvaluationError):
            b0.evaluate_prediction(gt, pred)

    def test_historical_csv_migration_and_replacement(self):
        import pandas as pd
        gt = self.write('gt.csv', ['A', 'B'])
        pred = self.write('pred.csv', ['A', 'B'])
        output = self.root/'summary.csv'
        output.write_text('Dataset,PA,GA,FGA\nold,0.5,0.4,0.3\nnew,0,0,0\n', encoding='utf-8')
        for _ in range(2):
            legacy.evaluate_metrics(gt, pred, 'new', output_csv_path=output)
        rows = pd.read_csv(output).set_index('Dataset')
        self.assertEqual(len(rows), 2)
        self.assertTrue(pd.isna(rows.loc['old', 'FTA']))
        self.assertEqual(rows.loc['new', 'FTA'], 1)
        self.assertEqual(rows.loc['new', 'Matched_Templates'], 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
