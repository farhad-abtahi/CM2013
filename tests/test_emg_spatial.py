"""Check calibration-only alignment and the eight-channel forearm ring."""
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
namespace = {'np': np, 'EMGNinaproTrack': EMGNinaproTrack}
for cell_id in ['emg-normalisation-preprocess', 'emg-spatial-preprocess']:
    cell = next(c for c in notebook['cells'] if c.get('id') == cell_id)
    exec(''.join(cell['source']), namespace)
prepare = namespace['prepare_channel_order']
track = namespace['spatial_track']


class TestEMGSpatial(unittest.TestCase):
    def recording(self, subject='S1'):
        anchor = [1, 2, 3, 4, 20, 6, 7, 8, 50, 60]
        other = [10] * 8 + [90, 90]
        values = np.array([anchor, other, anchor, other,
                           [9, 80, 7, 6, 5, 4, 3, 2, 1, 10],
                           [1, 2, 3, 4, 5, 6, 90, 8, 9, 10]], float)
        emg = np.repeat(values[:, None, :], 4, axis=1)
        return Recording(subject, 100, {'emg': emg},
                         np.array(['g01', 'g02', 'g01', 'g02', 'g01', 'g02']),
                         {'rep': np.array([1, 1, 2, 2, 3, 4])})

    def test_fixed_ring_shift_preserves_other_channels_and_metadata(self):
        original = self.recording()
        before = original.epochs['emg'].copy()
        ready = prepare([original])[0]
        self.assertEqual(ready.meta['ring_shift'], -4)
        rms = track.preprocess(ready, {'normalisation': 'rms', 'channel_order': 'original'})
        aligned = track.preprocess(ready, {'normalisation': 'rms', 'channel_order': 'align_g01'})
        np.testing.assert_array_equal(aligned.epochs['emg'],
                                      rms.epochs['emg'][:, :, [4, 5, 6, 7, 0, 1, 2, 3, 8, 9]])
        np.testing.assert_array_equal(aligned.epochs['emg'][:, :, 8:], rms.epochs['emg'][:, :, 8:])
        np.testing.assert_array_equal(original.epochs['emg'], before)
        np.testing.assert_array_equal(aligned.meta['rep'], [3, 4])
        np.testing.assert_array_equal(aligned.labels, original.labels[4:])
        self.assertEqual(aligned.group, original.group)

    def test_scored_signals_and_labels_cannot_change_shift(self):
        original = self.recording()
        changed = self.recording()
        changed.epochs['emg'][4:] *= np.arange(1, 11) * 1000
        changed.labels[4:] = 'g12'
        a, b = prepare([original, changed])
        self.assertEqual(a.meta['ring_shift'], b.meta['ring_shift'])
        np.testing.assert_array_equal(a.meta['channel_rms'], b.meta['channel_rms'])
        np.testing.assert_array_equal(a.meta['anchor_profile'], b.meta['anchor_profile'])

    def test_each_subject_is_aligned_from_its_own_calibration(self):
        a = self.recording()
        b = self.recording('S2')
        b.epochs['emg'][:, :, :8] = np.roll(b.epochs['emg'][:, :, :8], 2, axis=2)
        first, second = prepare([a, b])
        self.assertEqual(first.meta['ring_shift'], -4)
        self.assertEqual(second.meta['ring_shift'], -6)
        cfg = {'normalisation': 'rms', 'channel_order': 'align_g01'}
        np.testing.assert_array_equal(track.preprocess(first, cfg).epochs['emg'],
                                      track.preprocess(second, cfg).epochs['emg'])

    def test_missing_known_calibration_gesture_is_rejected(self):
        rec = self.recording()
        rec.labels[:4] = 'g02'
        with self.assertRaises(ValueError):
            prepare([rec])


if __name__ == '__main__':
    unittest.main()
