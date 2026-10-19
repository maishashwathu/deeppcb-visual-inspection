"""Small correctness tests for evaluation, pairing and inference boundaries."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.pipeline import box_iou, match_predictions, propose, register_pair, describe, score_predictions

class PipelineTests(unittest.TestCase):
    def test_iou_and_empty(self):
        self.assertAlmostEqual(box_iou([[0,0,10,10]], [[5,0,15,10]])[0,0], 1/3)
        self.assertEqual(box_iou([], []).shape, (0,0))

    def test_duplicates_and_wrong_class(self):
        gt = np.array([[0,0,10,10,1]])
        pp = [dict(box=[0,0,10,10], label=1, score=.9),
              dict(box=[0,0,10,10], label=1, score=.8),
              dict(box=[0,0,10,10], label=2, score=.7)]
        outcome, used = match_predictions(pp, gt)
        self.assertEqual([ok for _,ok in outcome], [True,False,False])
        self.assertEqual(used, {0})

    def test_clean_pair_and_known_component(self):
        clean = np.zeros((64,64), np.uint8)
        self.assertEqual(len(propose(clean, clean)[0]), 0)
        changed = clean.copy(); changed[20:25,30:35] = 255
        boxes, _, _ = propose(clean, changed, min_area=4, dilation=1, pad=0)
        np.testing.assert_array_equal(boxes, [[29,19,36,26]])
        self.assertTrue(np.isfinite(describe(clean, changed, boxes[0])).all())

    def test_registration_rejects_blank(self):
        a = np.zeros((64,64), np.uint8)
        _, _, result = register_pair(a, a)
        self.assertFalse(result['success'])

    def test_metric_perfect_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            annotation = Path(folder) / 'labels.txt'
            annotation.write_text('\n'.join(f'{i*20} 0 {i*20+10} 10 {i+1}' for i in range(6)))
            records = [dict(id='example', annotation=str(annotation))]
            pp = [dict(box=[i*20,0,i*20+10,10], label=i+1, score=.9) for i in range(6)]
            _, score = score_predictions(records, {'example':pp})
            self.assertEqual(score['mAP'], 1.)
            self.assertEqual(score['F1'], 1.)
            _, missing = score_predictions(records, {'example':[]})
            self.assertEqual(missing['FN'], 6)
            self.assertEqual(missing['mAP'], 0.)

if __name__ == '__main__':
    unittest.main()
