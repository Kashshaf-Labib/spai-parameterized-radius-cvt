import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

from spai.config import get_config
from spai.utils import auto_resume_helper, load_checkpoint, save_checkpoint


class TestPhaseTwoResume(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_config({"cfg": "configs/spai_cvt.yaml"})
        self.config.defrost()
        self.config.MODEL.RESUME = "resume_cvt.pth"
        self.config.TRAIN.START_EPOCH = 0
        self.config.freeze()
        self.logger = logging.getLogger("test-phase-two-resume")

    @staticmethod
    def _make_training_state():
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return model, optimizer, scheduler

    def test_restores_model_optimizer_scheduler_and_next_epoch(self) -> None:
        source_model, source_optimizer, source_scheduler = self._make_training_state()
        checkpoint = {
            "model": source_model.state_dict(),
            "optimizer": source_optimizer.state_dict(),
            "lr_scheduler": source_scheduler.state_dict(),
            "max_accuracy": 0.75,
            "epoch": 3,
            "config": self.config,
        }

        target_model = nn.Linear(3, 2)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
        target_scheduler = torch.optim.lr_scheduler.StepLR(
            target_optimizer, step_size=1
        )

        with patch("spai.utils.torch.load", return_value=checkpoint) as torch_load:
            max_accuracy = load_checkpoint(
                self.config,
                target_model,
                target_optimizer,
                target_scheduler,
                self.logger,
            )

        torch_load.assert_called_once_with(
            "resume_cvt.pth", map_location="cpu", weights_only=False
        )
        self.assertEqual(max_accuracy, 0.75)
        self.assertEqual(self.config.TRAIN.START_EPOCH, 4)
        self.assertEqual(
            target_scheduler.state_dict(), source_scheduler.state_dict()
        )
        self.assertTrue(target_optimizer.state_dict()["state"])
        for target, source in zip(
            target_model.parameters(), source_model.parameters()
        ):
            torch.testing.assert_close(target, source)

    def test_rejects_incompatible_cvt_resume_checkpoint(self) -> None:
        model, optimizer, scheduler = self._make_training_state()
        checkpoint = {
            "model": {},
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": 0,
            "config": self.config,
        }

        with patch("spai.utils.torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(
                RuntimeError, "not compatible with the configured CvT-SPAI"
            ):
                load_checkpoint(
                    self.config, model, optimizer, scheduler, self.logger
                )

    def test_rejects_checkpoint_without_training_state(self) -> None:
        model, optimizer, scheduler = self._make_training_state()
        with patch(
            "spai.utils.torch.load", return_value={"model": model.state_dict()}
        ):
            with self.assertRaisesRegex(
                RuntimeError, "missing required training state"
            ):
                load_checkpoint(
                    self.config, model, optimizer, scheduler, self.logger
                )

    def test_config_accepts_an_explicit_resume_path(self) -> None:
        config = get_config({
            "cfg": "configs/spai_cvt.yaml",
            "resume": "attached/ckpt_epoch_4.pth",
        })
        self.assertEqual(
            config.MODEL.RESUME, "attached/ckpt_epoch_4.pth"
        )


class TestCheckpointPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_config({"cfg": "configs/spai_cvt.yaml"})
        self.config.defrost()
        self.config.AMP_OPT_LEVEL = "O0"
        self.config.freeze()
        self.logger = logging.getLogger("test-checkpoint-persistence")
        self.model, self.optimizer, self.scheduler = (
            TestPhaseTwoResume._make_training_state()
        )

    def _set_output(self, output: str) -> None:
        self.config.defrost()
        self.config.OUTPUT = output
        self.config.freeze()

    def test_checkpoint_is_published_atomically(self) -> None:
        with TemporaryDirectory() as temp_dir:
            self._set_output(temp_dir)
            save_checkpoint(
                self.config,
                7,
                self.model,
                0.5,
                self.optimizer,
                self.scheduler,
                self.logger,
            )

            checkpoint_path = Path(temp_dir) / "ckpt_epoch_7.pth"
            self.assertTrue(checkpoint_path.is_file())
            self.assertFalse(Path(f"{checkpoint_path}.tmp").exists())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epoch"], 7)
            self.assertEqual(checkpoint["max_accuracy"], 0.5)

    def test_failed_save_preserves_previous_checkpoint_and_removes_temp(self) -> None:
        with TemporaryDirectory() as temp_dir:
            self._set_output(temp_dir)
            checkpoint_path = Path(temp_dir) / "ckpt_epoch_2.pth"
            checkpoint_path.write_bytes(b"previous-complete-checkpoint")

            def fail_after_partial_write(_state, temporary_path) -> None:
                Path(temporary_path).write_bytes(b"partial")
                raise RuntimeError("simulated interrupted write")

            with patch(
                "spai.utils.torch.save", side_effect=fail_after_partial_write
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted write"):
                    save_checkpoint(
                        self.config,
                        2,
                        self.model,
                        0.5,
                        self.optimizer,
                        self.scheduler,
                        self.logger,
                    )

            self.assertEqual(
                checkpoint_path.read_bytes(), b"previous-complete-checkpoint"
            )
            self.assertFalse(Path(f"{checkpoint_path}.tmp").exists())

    def test_auto_resume_selects_highest_valid_epoch(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            for filename in (
                "ckpt_epoch_2.pth",
                "ckpt_epoch_10.pth",
                "ckpt_epoch_invalid.pth",
                "ckpt_epoch_11.pth.tmp",
                "best.pth",
            ):
                (output_path / filename).write_bytes(b"test")

            selected = auto_resume_helper(output_path, self.logger)
            self.assertEqual(selected, str(output_path / "ckpt_epoch_10.pth"))


if __name__ == "__main__":
    unittest.main()
