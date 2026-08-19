import unittest

import torch
from torch import nn

from spai.config import get_config
from spai.models.cvt import build_cvt
from spai.models.sid import MFViT
from spai.utils import extract_mfm_encoder_state_dict


class _FirstFeature(nn.Module):
    def forward(self, original, low_frequency, high_frequency):
        return original


class TestCvtBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cvt_backbone = build_cvt(get_config({"cfg": "configs/spai.yaml"}))

    def test_cvt_13_structure_matches_mfm_checkpoint(self):
        config = self.cvt_backbone.encoder.config

        self.assertEqual(config.depth, [1, 2, 10])
        self.assertEqual(config.embed_dim, [64, 192, 384])
        self.assertEqual(config.num_heads, [1, 3, 6])
        self.assertEqual(self.cvt_backbone.num_features, 10)
        self.assertEqual(self.cvt_backbone.feature_dim, 384)
        self.assertEqual(self.cvt_backbone.encoder_stride, 16)

    def test_cvt_returns_homogeneous_final_stage_features(self):
        self.cvt_backbone.eval()
        with torch.inference_mode():
            features = self.cvt_backbone(torch.randn(1, 3, 32, 32))

        self.assertEqual(features.shape, (1, 10, 4, 384))

    def test_cvt_rejects_invalid_feature_layer(self):
        config = get_config({"cfg": "configs/spai.yaml"})
        config.defrost()
        config.MODEL.CVT.FEATURE_LAYERS = [10]
        config.freeze()

        with self.assertRaisesRegex(ValueError, "FEATURE_LAYERS"):
            build_cvt(config)

    def test_mfm_encoder_prefix_conversion_is_exact_and_strict(self):
        expected_state = self.cvt_backbone.state_dict()
        checkpoint_state = {
            f"encoder.{key}": value for key, value in expected_state.items()
        }
        checkpoint_state["decoder.0.weight"] = torch.empty(1)

        converted_state = extract_mfm_encoder_state_dict(checkpoint_state)

        self.assertEqual(set(converted_state), set(expected_state))
        self.assertNotIn("decoder.0.weight", converted_state)
        self.cvt_backbone.load_state_dict(converted_state, strict=True)

    def test_frozen_cvt_stays_in_eval_mode_during_training(self):
        model = MFViT(
            vit=self.cvt_backbone,
            features_processor=_FirstFeature(),
            cls_head=None,
            masking_radius=4,
            img_size=32,
            frozen_backbone=True,
            initialization_scope="local",
        )

        model.train()

        batch_norms = [
            module for module in self.cvt_backbone.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in self.cvt_backbone.parameters()
        ))
        self.assertTrue(all(not module.training for module in batch_norms))

        running_means = [module.running_mean.clone() for module in batch_norms]
        with torch.no_grad():
            model(torch.rand(1, 3, 32, 32))
        for before, module in zip(running_means, batch_norms):
            self.assertTrue(torch.equal(before, module.running_mean))
