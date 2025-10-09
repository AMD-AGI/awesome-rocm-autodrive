// Copyright (C) 2022 Cambricon.
//
// SPDX-License-Identifier: MIT
#ifndef PSAMASK_UTILS_HPP_
#define PSAMASK_UTILS_HPP_

typedef enum {
  COLLECT = 0,
  DISTRIBUTE = 1,
} PsamaskType;

typedef enum {
  PARTITION_N = 0,
  PARTITION_H = 1,
} DimPartitionType;

struct PartitionSeg {
  int h_per_cluster;
  int n_per_cluster;
  int h_per_core;
  int n_per_core;
  DimPartitionType cluster_partition;
  DimPartitionType core_partition;
};

struct Shape {
  int n;
  int h;
  int w;
  int c;
};

struct LimitParam {
  int n;
  int h;
  int w;
};

struct PositionInCore {
  int n_start;
  int n_end;
  int h_start;
  int h_end;
  int w_start;
  int w_end;
};
#endif  // PSAMASK_UTILS_HPP_
