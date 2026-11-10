"""Check that notebook calibration stays separate from scored repetitions."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tracks"), str(ROOT / "src")]
from adapter import Recording
from emg_ninapro import EMGNinaproTrack

notebook = json.loads((ROOT / "notebooks/track_emg_ninapro.ipynb").read_text())
source = next(c for c in notebook['cells'] if c.get('id') == 'emg-normalisation-preprocess')
namespace = {'np': np, 'EMGNinaproTrack': EMGNinaproTrack}
exec(''.join(source['source']), namespace)
prepare = namespace['prepare_normalisation']
track = namespace['normalisation_track']


class TestEMGNormalisation(unittest.TestCase):
    def recording(self, subject="S1", gain=(1, 1, 1)):
        # Same waveform in each repetition, with a deliberately flat third channel.
        window = np.array([[1, 2, 0], [2, 4, 0], [3, 6, 0], [4, 8, 0]], float)
        emg = np.tile(window, (6, 1, 1)) * np.asarray(gain)
        return Recording(subject, 100, {"emg": emg},
                         np.array(["g01"] * 6), {"rep": np.arange(1, 7)})

    def test_scoring_data_cannot_change_calibration(self):
        original = self.recording()
        changed = self.recording()
        changed.epochs['emg'][2:] *= 1000
        changed.labels[:] = 'g09'
        first = prepare([original])[0]
        second = prepare([changed])[0]
        np.testing.assert_array_equal(first.meta['channel_rms'], second.meta['channel_rms'])
        np.testing.assert_allclose(
            track.preprocess(second, {'normalisation': 'rms'}).epochs['emg'],
            track.preprocess(first, {'normalisation': 'rms'}).epochs['emg'] * 1000)
        np.testing.assert_array_equal(first.meta['rep'], [3, 4, 5, 6])
        np.testing.assert_array_equal(original.meta['rep'], [1, 2, 3, 4, 5, 6])
        self.assertEqual(original.epochs['emg'].shape[0], 6)

    def test_each_subject_uses_its_own_channel_scales(self):
        a, b = prepare([self.recording(), self.recording('S2', (3, 5, 7))])
        np.testing.assert_allclose(b.meta['channel_rms'][:2] / a.meta['channel_rms'][:2], [3, 5])
        result_a = track.preprocess(a, {'normalisation': 'rms'})
        result_b = track.preprocess(b, {'normalisation': 'rms'})
        np.testing.assert_allclose(result_a.epochs['emg'], result_b.epochs['emg'])
        self.assertTrue(np.isfinite(result_a.epochs['emg']).all())
        np.testing.assert_array_equal(result_a.epochs['emg'][:, :, 2], 0)
        self.assertEqual(result_a.group, a.group)
        np.testing.assert_array_equal(result_a.labels, a.labels)
        np.testing.assert_array_equal(result_a.meta['rep'], a.meta['rep'])
        self.assertIs(track.preprocess(a, {'normalisation': 'none'}), a)

    def test_missing_calibration_is_rejected(self):
        rec = self.recording()
        rec.meta['rep'] = np.arange(3, 9)
        with self.assertRaises(ValueError):
            prepare([rec])


if __name__ == '__main__':
    unittest.main()
