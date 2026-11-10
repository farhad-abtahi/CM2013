"""Check that the no-RMS experiment only changes channel order."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tracks'), str(ROOT / 'src')]
from adapter import Recording
from emg_ninapro import EMGNinaproTrack

notebook = json.loads((ROOT / 'notebooks/track_emg_ninapro.ipynb').read_text())
cell = next(c for c in notebook['cells'] if c.get('id') == 'emg-spatial-no-rms-preprocess')
namespace = {'np': np, 'EMGNinaproTrack': EMGNinaproTrack}
exec(''.join(cell['source']), namespace)
prepare = namespace['prepare_unscaled_alignment']
track = namespace['unscaled_track']


class TestUnscaledSpatial(unittest.TestCase):
    def recording(self, subject='S1'):
        anchor = [3, 4, 90, 6, 7, 8, 9, 10, 200, 300]
        other = [10] * 10
        values = np.array([anchor, other, anchor, other,
                           [1, 80, 3, 4, 5, 6, 7, 8, 9, 10],
                           [1, 2, 3, 4, 5, 6, 70, 8, 9, 10]], float)
        return Recording(subject, 100, {'emg': np.repeat(values[:, None, :], 4, axis=1)},
                         np.array(['g01', 'g02', 'g01', 'g02', 'g01', 'g02']),
                         {'rep': np.array([1, 1, 2, 2, 3, 4])})

    def test_original_and_shifted_values_are_not_scaled(self):
        original = self.recording()
        before = original.epochs['emg'].copy()
        rec = prepare([original])[0]
        self.assertNotIn('channel_rms', rec.meta)
        self.assertEqual(rec.meta['ring_shift'], -2)
        baseline = track.preprocess(rec, {'channel_order': 'original'})
        aligned = track.preprocess(rec, {'channel_order': 'align_g01'})
        np.testing.assert_array_equal(baseline.epochs['emg'], before[4:])
        expected = before[4:][:, :, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9]]
        np.testing.assert_array_equal(aligned.epochs['emg'], expected)
        np.testing.assert_array_equal(aligned.epochs['emg'][:, :, 8:], before[4:, :, 8:])
        np.testing.assert_array_equal(aligned.meta['rep'], [3, 4])
        np.testing.assert_array_equal(aligned.labels, original.labels[4:])
        np.testing.assert_array_equal(original.epochs['emg'], before)

    def test_scoring_data_cannot_change_shift(self):
        original = self.recording()
        changed = self.recording()
        changed.epochs['emg'][4:] *= np.arange(1, 11) * 1000
        changed.labels[4:] = ['g12', 'g01']
        a, b = prepare([original, changed])
        self.assertEqual(a.meta['ring_shift'], b.meta['ring_shift'])
        np.testing.assert_array_equal(a.meta['anchor_profile'], b.meta['anchor_profile'])

    def test_each_subject_uses_its_own_calibration(self):
        a = self.recording()
        b = self.recording('S2')
        b.epochs['emg'][:, :, :8] = np.roll(b.epochs['emg'][:, :, :8], 3, axis=2)
        first, second = prepare([a, b])
        self.assertEqual(first.meta['ring_shift'], -2)
        self.assertEqual(second.meta['ring_shift'], -5)
        np.testing.assert_array_equal(
            track.preprocess(first, {'channel_order': 'align_g01'}).epochs['emg'],
            track.preprocess(second, {'channel_order': 'align_g01'}).epochs['emg'])

    def test_missing_calibration_gesture_is_rejected(self):
        rec = self.recording()
        rec.labels[:4] = 'g02'
        with self.assertRaises(ValueError):
            prepare([rec])


if __name__ == '__main__':
    unittest.main()
