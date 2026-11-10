"""Check the relative-MAV calculation without changing other features."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tracks'), str(ROOT / 'src')]
from adapter import Recording
from emg_ninapro import EMGNinaproTrack

n = json.loads((ROOT / 'notebooks/track_emg_ninapro.ipynb').read_text())
namespace = {'np': np, 'EMGNinaproTrack': EMGNinaproTrack}
for cell_id in ['emg-normalisation-preprocess', 'emg-relative-mav-features']:
    c = next(c for c in n['cells'] if c.get('id') == cell_id)
    exec(''.join(c['source']), namespace)
track = namespace['feature_track']


class TestRelativeMAV(unittest.TestCase):
    def recording(self):
        signal = np.arange(1, 21)[:, None] * np.arange(1, 11)[None, :]
        return Recording('S1', 100, {'emg': np.array([signal, signal * 3], float)},
                         np.array(['g01', 'g02']), {'rep': np.array([3, 4])})

    def features(self, rec, method):
        return track._features_for(rec, {'normalisation': 'none', 'mav_feature': method})

    def test_shares_and_other_features(self):
        rec = self.recording()
        original = rec.epochs['emg'].copy()
        baseline, labels, group = self.features(rec, 'absolute')
        relative, new_labels, new_group = self.features(rec, 'relative')
        self.assertEqual(relative.shape, (2, 50))
        expected = np.arange(1, 11) / 55
        np.testing.assert_allclose(relative[:, 0::5], np.tile(expected, (2, 1)))
        np.testing.assert_allclose(relative[:, 0::5].sum(axis=1), 1)
        other_columns = np.arange(50) % 5 != 0
        np.testing.assert_array_equal(relative[:, other_columns], baseline[:, other_columns])
        np.testing.assert_array_equal(new_labels, labels)
        self.assertEqual(group, new_group)
        np.testing.assert_array_equal(rec.epochs['emg'], original)

    def test_window_gain_does_not_change_shares(self):
        original = self.recording()
        changed = self.recording()
        changed.epochs['emg'] *= np.array([2, 7])[:, None, None]
        first, _, _ = self.features(original, 'relative')
        second, _, _ = self.features(changed, 'relative')
        np.testing.assert_allclose(first[:, 0::5], second[:, 0::5])

    def test_zero_activity_stays_finite(self):
        rec = self.recording()
        rec.epochs['emg'][:] = 0
        features, _, _ = self.features(rec, 'relative')
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:, 0::5], 0)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            self.features(self.recording(), 'unknown')


if __name__ == '__main__':
    unittest.main()
