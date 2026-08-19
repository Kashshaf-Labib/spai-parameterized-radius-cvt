import unittest
import logging
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from spai.config import get_config
from spai.models.cvt import build_cvt
from spai.models.build import build_cls_model
from spai.models.sid import MFViT, PatchBasedMFViT
from spai.optimizer import build_optimizer, get_cvt_layer
from spai.utils import extract_mfm_encoder_state_dict, load_pretrained


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

    def test_phase_two_builder_uses_cvt_feature_dimensions(self):
        config = get_config({"cfg": "configs/spai_cvt.yaml"})

        model = build_cls_model(config)

        self.assertIsInstance(model, PatchBasedMFViT)
        self.assertIsInstance(model.get_vision_transformer(), type(self.cvt_backbone))
        self.assertEqual(model.cls_vector_dim, 1084)
        self.assertEqual(model.cls_head.head[0].in_features, 1084)

        features = torch.randn(1, 10, 4, 384)
        with torch.no_grad():
            spectral_vector = model.mfvit.features_processor(
                features, features, features
            )
        self.assertEqual(spectral_vector.shape, (1, 1084))

        phase_two_checkpoint = {"model": model.state_dict(), "epoch": 7}
        with patch("spai.utils.torch.load", return_value=phase_two_checkpoint):
            epoch = load_pretrained(
                config,
                model,
                logging.getLogger("test-cvt-phase-two-load"),
                checkpoint_path=Path("phase_two_cvt.pth"),
                verbose=False,
            )
        self.assertEqual(epoch, 7)

        incompatible_state = dict(model.state_dict())
        incompatible_state.pop(next(iter(incompatible_state)))
        with patch(
            "spai.utils.torch.load",
            return_value={"model": incompatible_state, "epoch": 7},
        ):
            with self.assertRaisesRegex(RuntimeError, "not compatible"):
                load_pretrained(
                    config,
                    model,
                    logging.getLogger("test-cvt-incompatible-load"),
                    checkpoint_path=Path("incompatible_cvt.pth"),
                    verbose=False,
                )

    def test_cvt_learnable_radius_config_inherits_cvt_settings(self):
        config = get_config({"cfg": "configs/spai_cvt_learnable_radius.yaml"})

        model = build_cls_model(config)

        self.assertEqual(config.MODEL.TYPE, "cvt")
        self.assertTrue(config.MODEL.FRE.LEARNABLE_MASKING_RADIUS)
        self.assertEqual(config.MODEL.CVT.FEATURE_LAYERS, list(range(10)))
        self.assertIsInstance(model, PatchBasedMFViT)
        self.assertTrue(model.mfvit.learnable_radius)
        self.assertEqual(model.cls_vector_dim, 1084)

        optimizer = build_optimizer(
            config, model, logging.getLogger("test-cvt-optimizer"), is_pretrain=False
        )
        optimized_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(all(
            id(parameter) not in optimized_parameters
            for parameter in model.get_vision_transformer().parameters()
        ))
        radius_groups = [
            group for group in optimizer.param_groups
            if group.get("group_name") == "masking_radius"
        ]
        self.assertEqual(len(radius_groups), 1)
        self.assertEqual(radius_groups[0]["lr"], config.TRAIN.RADIUS_LR)

    def test_cvt_layer_decay_mapping(self):
        depths = [1, 2, 10]
        num_layers = sum(depths) + 2

        self.assertEqual(get_cvt_layer(
            "mfvit.vit.encoder.encoder.stages.0.embedding.convolution_embeddings.projection.weight",
            num_layers,
            depths,
        ), 0)
        self.assertEqual(get_cvt_layer(
            "mfvit.vit.encoder.encoder.stages.2.layers.9.attention.attention.query.weight",
            num_layers,
            depths,
        ), 13)
        self.assertEqual(get_cvt_layer("cls_head.head.0.weight", num_layers, depths), 14)
