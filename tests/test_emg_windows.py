"""Check EMG window lengths and gesture/repetition boundaries."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tracks"), str(ROOT / "src")]
from emg_ninapro import EMGNinaproTrack


class TestEMGWindows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = Path(self.temp.name)
        self.signal = np.repeat(np.arange(60)[:, None], 10, axis=1)

    def tearDown(self):
        self.temp.cleanup()

    def write_recording(self, labels, repetitions):
        savemat(self.cache / "S1_A1_E1.mat", {
            "emg": self.signal,
            "restimulus": np.asarray(labels)[:, None],
            "rerepetition": np.asarray(repetitions)[:, None],
        })

    def test_lengths_and_default(self):
        self.write_recording([1] * 30 + [2] * 30, [1] * 30 + [2] * 30)
        track = EMGNinaproTrack()
        for milliseconds, samples, starts in [
            (100, 10, [0, 15, 30, 45]),
            (200, 20, [0, 30]),
            (300, 30, [0, 30]),
        ]:
            with self.subTest(window_ms=milliseconds):
                rec = track.load(self.cache, window_ms=milliseconds)[0]
                self.assertEqual(rec.epochs["emg"].shape, (len(starts), samples, 10))
                np.testing.assert_array_equal(rec.epochs["emg"][:, 0, 0], starts)
                for i, start in enumerate(starts):
                    np.testing.assert_array_equal(rec.epochs["emg"][i], self.signal[start:start + samples])
                    self.assertEqual(rec.labels[i], "g01" if start < 30 else "g02")
                    self.assertEqual(rec.meta["rep"][i], 1 if start < 30 else 2)
        default = track.load(self.cache)[0]
        explicit = track.load(self.cache, window_ms=200)[0]
        np.testing.assert_array_equal(default.epochs["emg"], explicit.epochs["emg"])

    def test_window_does_not_cross_repetitions(self):
        self.write_recording([1] * 60, [1] * 30 + [2] * 30)
        rec = EMGNinaproTrack().load(self.cache, window_ms=300)[0]
        np.testing.assert_array_equal(rec.epochs["emg"][:, 0, 0], [0, 30])
        np.testing.assert_array_equal(rec.meta["rep"], [1, 2])

    def test_invalid_lengths(self):
        for length in [0, -100, 5, 15]:
            with self.subTest(window_ms=length):
                with self.assertRaises(ValueError):
                    EMGNinaproTrack().load(self.cache, window_ms=length)


if __name__ == "__main__":
    unittest.main()
