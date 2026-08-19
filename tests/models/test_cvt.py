import unittest

import torch

from spai.config import get_config
from spai.models.cvt import build_cvt


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
