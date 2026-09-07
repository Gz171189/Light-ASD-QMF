"""Synthetic AVA preprocessing checks; no downloads or dataset needed."""

import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile

from utils.tools import extract_audio, extract_audio_clips, extract_video_clips


class PreprocessingChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ava_preprocess_check_')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.args = SimpleNamespace(**{name: str(self.root / directory) for name, directory in (
            ('trialPathAVA', 'csv'), ('audioOrigPathAVA', 'orig_audio'),
            ('audioPathAVA', 'audio'), ('visualOrigPathAVA', 'orig_video'),
            ('visualPathAVA', 'video'))})
        for path in vars(self.args).values():
            Path(path).mkdir()
        self.rows = [dict(video_id='video1', entity_id='entity1', frame_timestamp=t,
                          label_id=1, instance_id='instance1', entity_box_x1=0.1,
                          entity_box_y1=0.1, entity_box_x2=0.9, entity_box_y2=0.9)
                     for t in (0.1, 0.2)]
        self.write_annotations()
        self.original_audio = Path(self.args.audioOrigPathAVA) / 'trainval/video1.wav'
        self.original_audio.parent.mkdir()
        self.original_video = Path(self.args.visualOrigPathAVA) / 'trainval/video1.avi'
        self.original_video.parent.mkdir()
        self.audio_target = Path(self.args.audioPathAVA) / 'train/video1/entity1.wav'
        self.face_dir = Path(self.args.visualPathAVA) / 'train/video1/entity1'

    def write_annotations(self):
        for split in ('train', 'val', 'test'):
            pd.DataFrame(self.rows if split == 'train' else [],
                         columns=self.rows[0]).to_csv(
                             Path(self.args.trialPathAVA) / (split + '_orig.csv'), index=False)

    def run_quietly(self, function):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            function(self.args)
        return output.getvalue()

    def assert_context(self, exc, path):
        message = str(exc.exception)
        for value in ('split=train', 'video_id=video1', 'entity_id=entity1',
                      'timestamp=', str(path)):
            self.assertIn(value, message)

    def make_audio(self):
        self.samples = np.arange(16000, dtype=np.int16)
        wavfile.write(str(self.original_audio), 16000, self.samples)

    def make_video(self):
        writer = cv2.VideoWriter(str(self.original_video), cv2.VideoWriter_fourcc(*'MJPG'),
                                 25, (32, 32))
        self.assertTrue(writer.isOpened())
        try:
            for value in range(10):
                writer.write(np.full((32, 32, 3), value * 20, dtype=np.uint8))
        finally:
            writer.release()

    def test_audio_success_and_resume(self):
        self.make_audio()
        self.assertIn('processed=1, skipped_existing=0, failed=0',
                      self.run_quietly(extract_audio_clips))
        sr, samples = wavfile.read(str(self.audio_target))
        self.assertEqual(sr, 16000)
        np.testing.assert_array_equal(samples, self.samples[1600:3200])
        original = self.audio_target.read_bytes()
        self.original_audio.unlink()
        self.assertIn('processed=0, skipped_existing=1, failed=0',
                      self.run_quietly(extract_audio_clips))
        self.assertEqual(self.audio_target.read_bytes(), original)

    def test_missing_and_unreadable_audio(self):
        for corrupt in (False, True):
            if corrupt:
                self.original_audio.write_bytes(b'bad wav')
            with self.subTest(corrupt=corrupt), self.assertRaises(
                    RuntimeError if corrupt else FileNotFoundError) as exc:
                self.run_quietly(extract_audio_clips)
            self.assert_context(exc, self.original_audio)

    def test_audio_out_of_range(self):
        wavfile.write(str(self.original_audio), 16000, np.zeros(100, dtype=np.int16))
        with self.assertRaisesRegex(RuntimeError, 'out-of-range') as exc:
            self.run_quietly(extract_audio_clips)
        self.assert_context(exc, self.audio_target)

    def test_failed_audio_write_leaves_no_completed_target(self):
        self.make_audio()
        with patch('utils.tools.wavfile.write', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(RuntimeError, 'Failed to write') as exc:
                self.run_quietly(extract_audio_clips)
        self.assert_context(exc, self.audio_target)
        self.assertFalse(self.audio_target.exists())

    def test_video_success_partial_and_complete_resume(self):
        self.make_video()
        self.assertIn('processed=2, skipped_existing=0, failed=0',
                      self.run_quietly(extract_video_clips))
        first = self.face_dir / '0.10.jpg'
        second = self.face_dir / '0.20.jpg'
        saved = first.read_bytes()
        second.unlink()
        self.assertIn('processed=1, skipped_existing=1, failed=0',
                      self.run_quietly(extract_video_clips))
        self.assertEqual(first.read_bytes(), saved)
        self.original_video.unlink()
        self.assertIn('processed=0, skipped_existing=2, failed=0',
                      self.run_quietly(extract_video_clips))

    def test_missing_video(self):
        with self.assertRaises(FileNotFoundError) as exc:
            self.run_quietly(extract_video_clips)
        self.assert_context(exc, self.original_video.parent)

    def test_video_open_seek_and_read_failures_release_capture(self):
        self.original_video.touch()
        for failure in ('open', 'seek', 'read'):
            with self.subTest(failure=failure), patch('utils.tools.cv2.VideoCapture') as factory:
                capture = factory.return_value
                capture.isOpened.return_value = failure != 'open'
                capture.set.return_value = failure != 'seek'
                capture.read.return_value = (False, None)
                with self.assertRaises(RuntimeError) as exc:
                    self.run_quietly(extract_video_clips)
                self.assert_context(exc, self.original_video)
                capture.release.assert_called_once()

    def test_invalid_crop(self):
        self.make_video()
        self.rows[0]['entity_box_x2'] = 0.0
        self.write_annotations()
        with self.assertRaisesRegex(RuntimeError, 'Invalid AVA face crop') as exc:
            self.run_quietly(extract_video_clips)
        self.assert_context(exc, self.original_video)

    def test_failed_or_missing_image_write(self):
        self.make_video()
        for result in (False, True):
            with self.subTest(imwrite_result=result), patch('utils.tools.cv2.imwrite', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'Failed to write') as exc:
                    self.run_quietly(extract_video_clips)
                self.assert_context(exc, self.original_video)
                self.assertFalse((self.face_dir / '0.10.jpg').exists())

    def test_empty_existing_target_is_not_skipped(self):
        self.face_dir.mkdir(parents=True)
        (self.face_dir / '0.10.jpg').touch()
        with self.assertRaisesRegex(RuntimeError, 'Missing/empty'):
            self.run_quietly(extract_video_clips)

    def test_ffmpeg_failure_and_missing_output(self):
        self.original_video.touch()
        for failure in (subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'bad input'), None):
            with self.subTest(failure=failure), patch('utils.tools.subprocess.run', side_effect=failure):
                with self.assertRaises(RuntimeError) as exc:
                    self.run_quietly(extract_audio)
                self.assertIn('split=trainval', str(exc.exception))
                self.assertIn(str(self.original_video).replace('\\', '/'),
                              str(exc.exception).replace('\\', '/'))
                self.assertFalse(self.original_audio.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
