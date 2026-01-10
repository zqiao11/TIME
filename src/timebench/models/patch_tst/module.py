# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Adapted from gift_eval for TIME benchmark.

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.modules.lambda_layer import LambdaLayer
from gluonts.torch.distributions.output import PtArgProj
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import take_last, unsqueeze_expand, weighted_average


class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: torch.Tensor) -> torch.Tensor:
        """
        Features are not interleaved.

        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """
        `input_ids_shape` is expected to be [bsz x seqlen x ...].
        """
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class PatchTSTModel(nn.Module):
    """
    Module implementing the PatchTST model for forecasting as described in
    https://arxiv.org/abs/2211.14730 extended to be probabilistic.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    patch_len
        Length of the patch.
    stride
        Stride of the patch.
    padding_patch
        Padding of the patch.
    d_model
        Size of hidden layers in the Transformer encoder.
    nhead
        Number of attention heads in the Transformer encoder.
    dim_feedforward
        Size of hidden layers in the Transformer encoder.
    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    dropout
        Dropout probability in the Transformer encoder.
    activation
        Activation function in the Transformer encoder.
    norm_first
        Whether to apply normalization before or after the attention.
    num_encoder_layers
        Number of layers in the Transformer encoder.
    scaling
        Scaling parameter can be "mean", "std" or None.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_feat_dynamic_real: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_encoder_layers: int,
        scaling: str,
        distr_output=StudentTOutput(),
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.patch_len = patch_len
        self.stride = stride
        self.context_length = context_length
        self.d_model = d_model
        self.padding_patch = padding_patch
        self.distr_output = distr_output
        self.num_feat_dynamic_real = num_feat_dynamic_real

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.patch_num = int((context_length - patch_len) / stride + 1)

        if padding_patch == "end":  # can be modified to general case
            if self.stride + self.prediction_length < self.patch_len:
                padding = self.patch_len - self.prediction_length
            else:
                padding = self.stride
            self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
            self.patch_num += 1

        # project from `patch_len` + `num_feat_dynamic_real` x `patch_len` to d_model
        self.patch_proj = nn.Linear(
            patch_len + self.num_feat_dynamic_real * patch_len, d_model
        )

        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.patch_num, d_model
        )

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )

        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # Distribution parameters projection
        args_dim = distr_output.args_dim
        projected_args_dim = {k: v * prediction_length for k, v in args_dim.items()}
        domain_map = LambdaLayer(distr_output.domain_map)
        self.args_proj = PtArgProj(
            in_features=self.patch_num * d_model,
            args_dim=projected_args_dim,
            domain_map=domain_map,
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        if self.num_feat_dynamic_real > 0:
            input_spec_feat = {
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            }
        else:
            input_spec_feat = {}

        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                **input_spec_feat,
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        # do patching
        if self.padding_patch == "end":
            past_target_scaled = self.padding_patch_layer(past_target_scaled)
        past_target_patches = past_target_scaled.unfold(
            dimension=1, size=self.patch_len, step=self.stride
        )

        # do patching for time features as well
        if self.num_feat_dynamic_real > 0:
            # shift time features by `prediction_length` so that they are
            # aligned with the target input.
            time_feat = take_last(
                torch.cat((past_time_feat, future_time_feat), dim=1),
                dim=1,
                num=self.context_length,
            )

            # (bs x T x d) --> (bs x d x T) because the 1D padding is done on
            # the last dimension.
            time_feat = self.padding_patch_layer(
                time_feat.transpose(-2, -1)
            ).transpose(-2, -1)
            time_feat_patches = time_feat.unfold(
                dimension=1, size=self.patch_len, step=self.stride
            ).flatten(-2, -1)

        inputs = past_target_patches

        if self.num_feat_dynamic_real > 0:
            inputs = torch.cat((inputs, time_feat_patches), dim=-1)

        # project patches
        enc_in = self.patch_proj(inputs)
        embed_pos = self.positional_encoding(enc_in.size())

        # transformer encoder with positional encoding
        enc_out = self.encoder(enc_in + embed_pos)

        flattened_enc_out = enc_out.flatten(start_dim=1)

        # project to distribution arguments
        distr_args = self.args_proj(flattened_enc_out)

        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)

