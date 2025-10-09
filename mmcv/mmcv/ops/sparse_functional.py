# Copyright 2019 Yan Yan
#
# SPDX-License-Identifier: MIT
from typing import Any

import torch
from torch.autograd import Function

from . import sparse_ops as ops


class SparseConvFunction(Function):
    """Sparse Convolution.

    Please refer to `SECOND <https://www.mdpi.com/1424-8220/18/10/3337>`_ for
    more details.
    """

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False)

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, True, False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            True, False)

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, filters: torch.nn.Parameter,
                indice_pairs: torch.Tensor, indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(features, filters, indice_pairs,
                               indice_pair_num, num_activate_out, False, True)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            False, True)

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from sparse maxpooling.
        """
        out = ops.indice_maxpool(features, indice_pairs, indice_pair_num,
                                 num_activate_out)
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(features, out, grad_output,
                                               indice_pairs, indice_pair_num)
        return input_bp, None, None, None


indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
