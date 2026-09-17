"""No downloads: backend selection and both ASR return-value contracts."""
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.dotenv = patch.dict(sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda *a: None)})
        self.dotenv.start()

    def tearDown(self):
        self.dotenv.stop()
        self.environ.stop()

    def config(self, system="linux", machine="x86_64"):
        with patch("sys.platform", system), patch("platform.machine", return_value=machine):
            from app import config
            return importlib.reload(config)

    def test_linux_auto(self):
        c = self.config()
        self.assertEqual(c.ASR_BACKEND, "faster-whisper")
        self.assertEqual(c.ASR_MODEL, "turbo")
        self.assertEqual((c.ASR_DEVICE, c.ASR_COMPUTE_TYPE), ("cpu", "int8"))

    def test_mac_auto(self):
        c = self.config("darwin", "arm64")
        self.assertEqual(c.ASR_BACKEND, "mlx")
        self.assertEqual(c.ASR_MODEL, "mlx-community/whisper-large-v3-turbo")

    def test_linux_rejects_mlx(self):
        os.environ["EMOLABEL_ASR_BACKEND"] = "mlx"
        with self.assertRaisesRegex(ValueError, "Apple Silicon"):
            self.config()

    def test_invalid_backend(self):
        os.environ["EMOLABEL_ASR_BACKEND"] = "bogus"
        with self.assertRaisesRegex(ValueError, "ASR_BACKEND"):
            self.config()

    def test_binary_overrides(self):
        os.environ["EMOLABEL_FFMPEG"] = "/custom/ffmpeg"
        self.assertEqual(self.config().FFMPEG, "/custom/ffmpeg")

    def test_linux_transcription_schema_and_cache(self):
        self.config()
        from app import asr, asr_faster
        importlib.reload(asr_faster)
        model = MagicMock()
        seen = []

        def infer(*args, **kwargs):
            seen.append(kwargs)
            return iter([
                SimpleNamespace(start=0, end=1.23456, text=" 안녕! ", no_speech_prob=0.1),
                SimpleNamespace(start=2, end=3, text="환각", no_speech_prob=0.9),
            ]), None

        model.transcribe.side_effect = infer
        factory = MagicMock(return_value=model)
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=factory)}):
            first = asr.transcribe(Path("fake.wav"))
            asr.transcribe(Path("fake2.wav"))
        self.assertEqual(first["engine"], "faster-whisper")
        self.assertEqual(first["text"], "안녕!")
        self.assertEqual(len(first["segments"]), 2)
        self.assertEqual(first["segments"][0]["end"], 1.235)
        self.assertFalse(seen[0]["condition_on_previous_text"])
        self.assertEqual(seen[0]["language"], "ko")
        factory.assert_called_once()

    def test_mac_preserves_schema(self):
        self.config("darwin", "arm64")
        from app import asr
        importlib.reload(asr)
        transcribe = MagicMock(return_value={"segments": [
            {"start": 0, "end": 1, "text": " 좋아. ", "no_speech_prob": 0.0}]})
        with patch.dict(sys.modules, {"mlx_whisper": SimpleNamespace(transcribe=transcribe)}):
            result = asr.transcribe(Path("fake.wav"))
        self.assertEqual(result["engine"], "mlx-whisper")
        self.assertEqual(result["text"], "좋아.")
        self.assertEqual(result["language"], "ko")


if __name__ == "__main__":
    unittest.main()
