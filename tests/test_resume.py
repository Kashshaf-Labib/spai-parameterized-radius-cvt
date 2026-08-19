import logging
import unittest
from unittest.mock import patch

import torch
from torch import nn

from spai.config import get_config
from spai.utils import load_checkpoint


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


if __name__ == "__main__":
    unittest.main()
