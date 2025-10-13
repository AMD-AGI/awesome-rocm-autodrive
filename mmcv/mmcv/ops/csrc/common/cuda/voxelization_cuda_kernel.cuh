// Copyright (c) OpenMMLab. All rights reserved.
#ifndef VOXELIZATION_CUDA_KERNEL_CUH
#define VOXELIZATION_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floorf((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      continue;
    }

    int c_y = floorf((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      continue;
    }

    int c_z = floorf((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
struct PointCoord {
    T_int x, y, z;
    int original_idx;

    __host__ __device__
    bool operator<(const PointCoord& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        if (z != other.z) return z < other.z;
        return original_idx < other.original_idx;  // stable sort
    }
    __host__ __device__
    bool operator==(const PointCoord& other) const {
      return x == other.x && y == other.y && z == other.z;
    }
};

// functor to avoid device lambda which need extra --extended-lambda flag in nvcc
template <typename T_int>
struct CoordTransformer {
    const T_int* coor_ptr;
    
    __host__ __device__
    PointCoord<T_int> operator()(int idx) const {
        PointCoord<int> pc;
        pc.x = coor_ptr[idx * 3];
        pc.y = coor_ptr[idx * 3 + 1];
        pc.z = coor_ptr[idx * 3 + 2];
        pc.original_idx = idx;
        return pc;
    }
};

template <typename T_int>
__global__ void point_to_voxelidx_with_sort_kernel(
    const PointCoord<T_int>* __restrict__ sorted_coords,
    T_int* __restrict__ point_to_voxelidx,
    T_int* __restrict__ point_to_pointidx,
    const int max_points,
    const int max_voxels,
    const int num_points) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= num_points) return;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
      auto current_coord = sorted_coords[index];
      if (current_coord.x == -1) continue;

      int equal_start = index;
      while (equal_start > 0 && sorted_coords[equal_start - 1] == current_coord) {
        equal_start--;
      }

      int pos_in_voxel = index - equal_start;
      T_int original_idx = current_coord.original_idx;

      if (pos_in_voxel < max_points) {
        point_to_pointidx[original_idx] = sorted_coords[equal_start].original_idx;
        point_to_voxelidx[original_idx] = pos_in_voxel;
      }
    }
}

template <typename T_int, int BLOCK_SIZE>
__global__ void point_to_voxelidx_kernel(const T_int* __restrict__ coor,
                                         T_int* __restrict__ point_to_voxelidx,
                                         T_int* __restrict__ point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  // __shared__ extern T_int smem_coor[];
  // T_int* shared_coor = smem_coor;
  // printf("max_points = %d, max_voxels = %d, num_points = %d, NDim = %d\n", max_points, max_voxels, num_points, NDim);

  struct __align__(16) Coor
  {
    T_int x, y, z;
    T_int pad;
  };
  __shared__ Coor shared_coor[BLOCK_SIZE];

  constexpr uint32_t elements_in_128b = 16 / sizeof(T_int);
  union BLOCK_16B
  {
    T_int e[elements_in_128b];
      __uint128_t ow;
  };

  int global_loop_cnt = (num_points + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int global_idx = 0; global_idx < global_loop_cnt; global_idx++) {
    bool is_valid = false;
    int num = 0;
    int first_match_idx = index;
    T_int coor_x = -1;
    T_int coor_y = -1;
    T_int coor_z = -1;

    if (index < num_points) {
      auto coor_offset = coor + index * NDim;
      // skip invalid points
      coor_x = __ldg(&coor_offset[0]);
      is_valid = (coor_x != -1);
      coor_y = __ldg(&coor_offset[1]);
      coor_z = __ldg(&coor_offset[2]);
    }

#pragma unroll
    for (int block_start = 0; block_start < num_points; block_start += BLOCK_SIZE) {
      // load coor to shared buffer
      // if (index >= block_start) {
        int load_pos = block_start + threadIdx.x;
        if (load_pos < num_points) {
          auto prev_coor = coor + load_pos * NDim;
          shared_coor[threadIdx.x].x = __ldg(&prev_coor[0]);
          shared_coor[threadIdx.x].y = __ldg(&prev_coor[1]);
          shared_coor[threadIdx.x].z = __ldg(&prev_coor[2]);
        }
      // }
      __syncthreads();

      // only calculate the coors before this coor[index]
      // if (is_valid && index < num_points) {
      if (is_valid) {
        BLOCK_16B v_ptr;
        // int block_end = min(block_start + BLOCK_SIZE, index);
        int block_end = min(min(block_start + BLOCK_SIZE, num_points), index);
#pragma unroll
        for (int i  = 0; i < block_end - block_start; i++) {
          // Find all previous points that have the same coors
          // if find the same coor, record it
          v_ptr.ow = *((const __uint128_t*)(shared_coor + i));
          bool is_match = (v_ptr.e[0] == coor_x) && (v_ptr.e[1] == coor_y) &&
                            (v_ptr.e[2] == coor_z);
          num += is_match ? 1 : 0;
          if (is_match && num == 1) {
            first_match_idx = block_start + i;
          } else if (is_match && num >= max_points) {
            // out of boundary
            break;
          }
        }
      }
      __syncthreads();
    }

    if (is_valid && index < num_points) {
      point_to_pointidx[index] = first_match_idx;
      if (num < max_points) {
        point_to_voxelidx[index] = num;
      }
    }

    index += blockDim.x * gridDim.x;
  }
}

template <typename T_int>
__global__ void mark_new_voxels(const T_int* point_to_voxelidx, bool* need_new_voxel,
                                  const int num_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_points) return;
  need_new_voxel[i] = (point_to_voxelidx[i] == 0);
}

template <typename T_int>
__global__ void update_voxels(T_int* num_points_per_voxel, T_int* coor_to_voxelidx,
                                const T_int* point_to_voxelidx, const int* voxel_indices,
                                const int max_voxels, const int num_points) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    int point_pos_in_voxel = point_to_voxelidx[index];
    if (point_pos_in_voxel == -1) continue;
    if (point_pos_in_voxel == 0) {
      int voxelidx = voxel_indices[index];
      if (voxelidx < max_voxels) {
        coor_to_voxelidx[index] = voxelidx;
        num_points_per_voxel[voxelidx] = 1;
      }
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx,
    const int* voxel_indices,
    const int max_points, const int max_voxels, const int num_points) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    int point_pos_in_voxel = point_to_voxelidx[index];
    if (point_pos_in_voxel == -1 || point_pos_in_voxel == 0) continue;

    int point_idx = point_to_pointidx[index];
    int voxelidx = coor_to_voxelidx[point_idx];
    if (voxelidx != -1) {
      coor_to_voxelidx[index] = voxelidx;
      atomicAdd(&num_points_per_voxel[voxelidx], 1);
    }
  }
}

__global__ void nondeterministic_get_assign_pos(
    const int nthreads, const int32_t* coors_map, int32_t* pts_id,
    int32_t* coors_count, int32_t* reduce_count, int32_t* coors_order) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    if (coors_idx > -1) {
      int32_t coors_pts_pos = atomicAdd(&reduce_count[coors_idx], 1);
      pts_id[thread_idx] = coors_pts_pos;
      if (coors_pts_pos == 0) {
        coors_order[coors_idx] = atomicAdd(coors_count, 1);
      }
    }
  }
}

template <typename T>
__global__ void nondeterministic_assign_point_voxel(
    const int nthreads, const T* points, const int32_t* coors_map,
    const int32_t* pts_id, const int32_t* coors_in, const int32_t* reduce_count,
    const int32_t* coors_order, T* voxels, int32_t* coors, int32_t* pts_count,
    const int max_voxels, const int max_points, const int num_features,
    const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    int coors_pts_pos = pts_id[thread_idx];
    if (coors_idx > -1 && coors_pts_pos < max_points) {
      int coors_pos = coors_order[coors_idx];
      if (coors_pos < max_voxels) {
        auto voxels_offset =
            voxels + (coors_pos * max_points + coors_pts_pos) * num_features;
        auto points_offset = points + thread_idx * num_features;
        for (int k = 0; k < num_features; k++) {
          voxels_offset[k] = points_offset[k];
        }
        if (coors_pts_pos == 0) {
          pts_count[coors_pos] = min(reduce_count[coors_idx], max_points);
          auto coors_offset = coors + coors_pos * NDim;
          auto coors_in_offset = coors_in + coors_idx * NDim;
          for (int k = 0; k < NDim; k++) {
            coors_offset[k] = coors_in_offset[k];
          }
        }
      }
    }
  }
}

#endif  // VOXELIZATION_CUDA_KERNEL_CUH
