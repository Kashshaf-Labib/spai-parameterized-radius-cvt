# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CvT-13 feature backbone used by the SPAI phase-two model."""

from typing import List

import torch
from torch import nn

try:
    from transformers import CvtConfig, CvtModel
except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
    raise ImportError(
        "The CvT backbone requires the 'transformers' package. "
        "Install the repository requirements before selecting MODEL.TYPE='cvt'."
    ) from exc


class CvtBackbone(nn.Module):
    """Expose homogeneous per-block features from one CvT encoder stage.

    SPAI compares original, low-frequency, and high-frequency features block by
    block. CvT's stages have different resolutions and channel widths, so the
    initial integration uses all ten blocks of its final stage. At 224x224 this
    returns ``[batch, 10, 196, 384]``, matching SPAI's existing feature layout.

    The wrapped model is deliberately named ``encoder``. The phase-one MFM
    checkpoint stores the same wrapper below its own ``encoder`` prefix, which
    permits a strict and auditable one-prefix conversion when weights are loaded.
    """

    def __init__(self, config):
        super().__init__()
        cvt = config.MODEL.CVT
        self.feature_stage = int(cvt.FEATURE_STAGE)
        self.feature_layers = tuple(int(index) for index in cvt.FEATURE_LAYERS)
        self._validate_config_feature_selection(cvt)

        transformer_config = CvtConfig(
            num_channels=cvt.IN_CHANS,
            patch_sizes=list(cvt.PATCH_SIZES),
            patch_stride=list(cvt.PATCH_STRIDES),
            patch_padding=list(cvt.PATCH_PADDINGS),
            embed_dim=list(cvt.EMBED_DIMS),
            depth=list(cvt.DEPTHS),
            num_heads=list(cvt.NUM_HEADS),
            mlp_ratio=list(cvt.MLP_RATIOS),
            qkv_bias=list(cvt.QKV_BIAS),
            qkv_projection_method=list(cvt.QKV_PROJECTION_METHOD),
            kernel_qkv=list(cvt.KERNEL_QKV),
            padding_q=list(cvt.PADDING_Q),
            padding_kv=list(cvt.PADDING_KV),
            stride_q=list(cvt.STRIDE_Q),
            stride_kv=list(cvt.STRIDE_KV),
            cls_token=list(cvt.CLS_TOKEN),
            drop_rate=list(cvt.DROP_RATE),
            attention_drop_rate=list(cvt.ATTENTION_DROP_RATE),
            drop_path_rate=list(cvt.DROP_PATH_RATE),
        )
        self.encoder = CvtModel(transformer_config)
        self._validate_feature_selection()

        self.num_features = len(self.feature_layers)
        self.feature_dim = int(cvt.EMBED_DIMS[self.feature_stage])
        self.encoder_stride = int(cvt.ENCODER_STRIDE)

    def _validate_config_feature_selection(self, cvt) -> None:
        if not 0 <= self.feature_stage < len(cvt.DEPTHS):
            raise ValueError(
                f"CVT.FEATURE_STAGE={self.feature_stage} is outside the "
                f"available range [0, {len(cvt.DEPTHS) - 1}]"
            )
        stage_depth = int(cvt.DEPTHS[self.feature_stage])
        self._validate_feature_layer_indices(stage_depth)

    def _validate_feature_selection(self) -> None:
        stages = self.encoder.encoder.stages
        if not 0 <= self.feature_stage < len(stages):
            raise ValueError(
                f"CVT.FEATURE_STAGE={self.feature_stage} is outside the "
                f"available range [0, {len(stages) - 1}]"
            )
        stage_depth = len(stages[self.feature_stage].layers)
        self._validate_feature_layer_indices(stage_depth)

    def _validate_feature_layer_indices(self, stage_depth: int) -> None:
        if not self.feature_layers:
            raise ValueError("CVT.FEATURE_LAYERS must select at least one layer")
        if len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError("CVT.FEATURE_LAYERS cannot contain duplicates")
        if tuple(sorted(self.feature_layers)) != self.feature_layers:
            raise ValueError("CVT.FEATURE_LAYERS must be in ascending order")
        if self.feature_layers[0] < 0 or self.feature_layers[-1] >= stage_depth:
            raise ValueError(
                f"CVT.FEATURE_LAYERS must be within [0, {stage_depth - 1}] for "
                f"stage {self.feature_stage}"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_state = pixel_values
        for stage_index, stage in enumerate(self.encoder.encoder.stages):
            if stage_index != self.feature_stage:
                hidden_state, _ = stage(hidden_state)
                continue

            return self._forward_feature_stage(stage, hidden_state)

        raise RuntimeError("The configured CvT feature stage was not reached")

    def _forward_feature_stage(self, stage: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = stage.embedding(hidden_state)
        batch_size, channels, height, width = hidden_state.shape
        hidden_state = hidden_state.view(batch_size, channels, height * width).permute(0, 2, 1)

        has_cls_token = bool(stage.config.cls_token[stage.stage])
        if has_cls_token:
            hidden_state = torch.cat(
                (stage.cls_token.expand(batch_size, -1, -1), hidden_state), dim=1
            )

        selected = set(self.feature_layers)
        features: List[torch.Tensor] = []
        for layer_index, layer in enumerate(stage.layers):
            hidden_state = layer(hidden_state, height, width)
            if layer_index in selected:
                features.append(hidden_state[:, 1:] if has_cls_token else hidden_state)

        return torch.stack(features, dim=1)


def build_cvt(config) -> CvtBackbone:
    """Build the CvT feature adapter without downloading external weights."""

    if config.MODEL.CVT.NAME != "cvt_13":
        raise ValueError(
            f"Unsupported CvT architecture '{config.MODEL.CVT.NAME}'; expected 'cvt_13'"
        )
    return CvtBackbone(config)
