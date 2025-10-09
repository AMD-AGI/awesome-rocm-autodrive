// Copyright 2019 Yan Yan
//
// SPDX-License-Identifier: MIT

#include <torch/script.h>
#include <utils/spconv/spconv/reordering.h>

#include "pytorch_cpp_helper.hpp"

namespace functor {
template <typename scalar_t, typename Index>
struct SparseGatherFunctor<tv::CPU, scalar_t, Index> {
  void operator()(const tv::CPU& d, tv::TensorView<scalar_t> buffer,
                  tv::TensorView<const scalar_t> features,
                  tv::TensorView<const Index> indices, int size) {
    int numPlanes = features.dim(1);
    for (int i = 0; i < size; ++i) {
      std::memcpy(buffer.data() + i * numPlanes,
                  features.data() + indices[i] * numPlanes,
                  sizeof(scalar_t) * numPlanes);
    }
  }
};

template <typename scalar_t, typename Index>
struct SparseScatterAddFunctor<tv::CPU, scalar_t, Index> {
  void operator()(const tv::CPU& d, tv::TensorView<scalar_t> outFeatures,
                  tv::TensorView<const scalar_t> buffer,
                  tv::TensorView<const Index> indices, int size, bool stable) {
    int numPlanes = outFeatures.dim(1);
    const scalar_t* buf = buffer.data();
    scalar_t* out = outFeatures.data();
    for (int i = 0; i < size; ++i) {
      buf = buffer.data() + i * numPlanes;
      out = outFeatures.data() + indices[i] * numPlanes;
      for (int j = 0; j < numPlanes; ++j) {
        out[j] += buf[j];
      }
    }
  }
};

}  // namespace functor

#define DECLARE_CPU_SPECS_T_INDEX(scalar_t, Index)                        \
  template struct functor::SparseGatherFunctor<tv::CPU, scalar_t, Index>; \
  template struct functor::SparseScatterAddFunctor<tv::CPU, scalar_t, Index>;

#define DECLARE_CPU_SPECS(scalar_t)         \
  DECLARE_CPU_SPECS_T_INDEX(scalar_t, int); \
  DECLARE_CPU_SPECS_T_INDEX(scalar_t, long);

DECLARE_CPU_SPECS(float);
DECLARE_CPU_SPECS(double);
DECLARE_CPU_SPECS(at::Half);

#undef DECLARE_CPU_SPECS
#undef DECLARE_CPU_SPECS_T_INDEX
