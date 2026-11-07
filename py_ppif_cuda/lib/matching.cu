#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>

#include "basic_types.h"

#define BLOCK_DIM 16

// Pivots of I - C^T C below this value are treated as zero, i.e. directions closer
// than ~0.2 degrees are treated as shared. Same as the CPU implementation.
#define SHARED_DIRECTION_TOLERANCE 1e-5f

/// Solve the NxN symmetric positive semi-definite system K x = z (with z in the
/// range of K) using a pivoted Cholesky factorisation K = P L L^T P^T. Pivots below
/// SHARED_DIRECTION_TOLERANCE are treated as zero and the matching entries of x are
/// set to 0, so a singular K gives a valid solution instead of dividing by ~0.
/// Same algorithm as solve_psd_system in py_ppif/lib/subspace_to_subspace_matching.cc.
///
/// @param K  NxN row-major matrix, overwritten (its lower triangle holds L).
/// @param z  N right-hand side.
/// @param x  N solution.
template<int N> __device__ void solve_psd_system(ScalarType* K, const ScalarType* z, ScalarType* x) {
  int perm[N];
  for (int i = 0; i < N; ++i) {
    perm[i] = i;
  }

  int rank = 0;
  for (int k = 0; k < N; ++k) {
    // Pivot on the largest remaining diagonal entry.
    int p = k;
    for (int i = k + 1; i < N; ++i) {
      if (K[i * N + i] > K[p * N + p]) {
        p = i;
      }
    }
    if (K[p * N + p] <= SHARED_DIRECTION_TOLERANCE) {
      break;
    }
    if (p != k) {
      for (int j = 0; j < N; ++j) {
        const ScalarType tmp = K[k * N + j];
        K[k * N + j] = K[p * N + j];
        K[p * N + j] = tmp;
      }
      for (int i = 0; i < N; ++i) {
        const ScalarType tmp = K[i * N + k];
        K[i * N + k] = K[i * N + p];
        K[i * N + p] = tmp;
      }
      const int tmp = perm[k];
      perm[k] = perm[p];
      perm[p] = tmp;
    }
    // Column k of L, then update the trailing block (kept symmetric so that
    // later row / column swaps stay valid).
    K[k * N + k] = sqrtf(K[k * N + k]);
    for (int i = k + 1; i < N; ++i) {
      K[i * N + k] /= K[k * N + k];
    }
    for (int i = k + 1; i < N; ++i) {
      for (int j = k + 1; j < N; ++j) {
        K[i * N + j] -= K[i * N + k] * K[j * N + k];
      }
    }
    ++rank;
  }

  // Solve L L^T w = P^T z on the leading rank x rank block.
  ScalarType w[N];
  for (int i = 0; i < rank; ++i) {
    ScalarType s = z[perm[i]];
    for (int j = 0; j < i; ++j) {
      s -= K[i * N + j] * w[j];
    }
    w[i] = s / K[i * N + i];
  }
  for (int i = rank - 1; i >= 0; --i) {
    ScalarType s = w[i];
    for (int j = i + 1; j < rank; ++j) {
      s -= K[j * N + i] * w[j];
    }
    w[i] = s / K[i * N + i];
  }

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }
  for (int i = 0; i < rank; ++i) {
    x[perm[i]] = w[i];
  }
}

// Subspace-to-subspace distance CUDA implementation.
template<int LIFTING_DIM> __global__ void compute_distance_between_two_affine_subspaces_kernel(
    const int n_A, const ScalarType* A, const int n_B,
    const ScalarType* B, ScalarType* distances) {
  const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx1 >= n_A || idx2 >= n_B) {
    return;
  }

// Element d of row i of subspace idx1 (A) / idx2 (B). Rows 0..LIFTING_DIM-1 are the
// orthonormal basis (D and E in the paper) and row LIFTING_DIM is the offset.
#define A_AT(i, d) A[(d) * (LIFTING_DIM + 1) * n_A + (i) * n_A + idx1]
#define B_AT(i, d) B[(d) * (LIFTING_DIM + 1) * n_B + (i) * n_B + idx2]

  // Same derivation as the CPU implementation: we look for
  // pA = cA + alpha^T bA and pB = cB + beta^T bB minimising ||pB - pA||.
  // Eliminating alpha = yA + C beta, with C = bA bB^T, yA = bA cAB and
  // yB = bB cAB, leaves the LIFTING_DIM x LIFTING_DIM system
  //   (I - C^T C) beta = -(yB - C^T yA).
  // I - C^T C is singular exactly when the subspaces share a direction (e.g. two
  // features lifted with the same two adversarial database words), where the
  // paper's block inversion of N = I - M M^T divides by ~0. Solving it as a
  // semi-definite system keeps the distance correct in that case.

  // Set up yA, yB.
  ScalarType yA[LIFTING_DIM];
  ScalarType yB[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    yA[i] = 0;
    yB[i] = 0;
  }
  for (int d = 0; d < DIM; ++d) {
    const ScalarType cAB = B_AT(LIFTING_DIM, d) - A_AT(LIFTING_DIM, d);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      yA[i] += A_AT(i, d) * cAB;
      yB[i] += B_AT(i, d) * cAB;
    }
  }

  // Set up C = bA bB^T.
  ScalarType C[LIFTING_DIM * LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    for (int j = 0; j < LIFTING_DIM; ++j) {
      ScalarType c = 0;
      for (int d = 0; d < DIM; ++d) {
        c += A_AT(i, d) * B_AT(j, d);
      }
      C[i * LIFTING_DIM + j] = c;
    }
  }

  // Set up K = I - C^T C and z = yB - C^T yA.
  ScalarType K[LIFTING_DIM * LIFTING_DIM];
  ScalarType z[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    z[i] = yB[i];
    for (int k = 0; k < LIFTING_DIM; ++k) {
      z[i] -= C[k * LIFTING_DIM + i] * yA[k];
    }
    for (int j = 0; j < LIFTING_DIM; ++j) {
      ScalarType v = (i == j) ? 1.0f : 0.0f;
      for (int k = 0; k < LIFTING_DIM; ++k) {
        v -= C[k * LIFTING_DIM + i] * C[k * LIFTING_DIM + j];
      }
      K[i * LIFTING_DIM + j] = v;
    }
  }

  // Solve for beta, then alpha = yA + C beta.
  ScalarType beta[LIFTING_DIM];
  solve_psd_system<LIFTING_DIM>(K, z, beta);
  for (int i = 0; i < LIFTING_DIM; ++i) {
    beta[i] = -beta[i];
  }
  ScalarType alpha[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    alpha[i] = yA[i];
    for (int j = 0; j < LIFTING_DIM; ++j) {
      alpha[i] += C[i * LIFTING_DIM + j] * beta[j];
    }
  }

  // Save the squared distance between pB and pA.
  ScalarType dist = 0;
  for (int d = 0; d < DIM; ++d) {
    ScalarType aux = B_AT(LIFTING_DIM, d) - A_AT(LIFTING_DIM, d);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      aux += beta[i] * B_AT(i, d) - alpha[i] * A_AT(i, d);
    }
    dist += aux * aux;
  }
  distances[idx1 * n_B + idx2] = dist;

#undef A_AT
#undef B_AT
}

template<int LIFTING_DIM> void compute_distances_between_affine_subspaces(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances) {
  assert(dim == DIM);
  assert(subspace_dim == LIFTING_DIM);

  // CUDA.
  cudaError_t cuda_status = cudaSuccess;

  // Allocate memory for distances.
  ScalarType* distances_device;
  cuda_status =
      cudaMalloc((void**)&distances_device, sizeof(ScalarType) * n_A * n_B);
  assert(cuda_status == cudaSuccess);

  // Run the kernel.
  const dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  const int n_blocks_A = (n_A + BLOCK_DIM - 1) / BLOCK_DIM;
  const int n_blocks_B = (n_B + BLOCK_DIM - 1) / BLOCK_DIM;
  const dim3 grid_dim(n_blocks_A, n_blocks_B);
  compute_distance_between_two_affine_subspaces_kernel<LIFTING_DIM><<<grid_dim, block_dim>>>(
      n_A, A_device, n_B, B_device, distances_device);
  // Check for errors.
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }
  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }

  // Copy from device.
  cuda_status =
      cudaMemcpy(distances, distances_device, sizeof(ScalarType) * n_A * n_B,
                 cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);

  // Free memory.
  cudaFree(distances_device);
}

template void compute_distances_between_affine_subspaces<2>(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances);

template void compute_distances_between_affine_subspaces<4>(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances);


// Point-to-subspace distance CUDA implementation.
template<int LIFTING_DIM> __global__ void compute_distance_between_affine_subspace_and_point_kernel(
    const int n_A, const ScalarType* A, const int n_B,
    const ScalarType* B, ScalarType* distances) {
  const int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx1 >= n_A || idx2 >= n_B) {
    return;
  }

  // Project point to subspace.
  ScalarType projection[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    projection[i] = 0;
    for (int d = 0; d < DIM; ++d) {
      projection[i] +=
          (B[d * n_B + idx2] -
           A[d * (LIFTING_DIM + 1) * n_A + LIFTING_DIM * n_A + idx1]) *
          A[d * (LIFTING_DIM + 1) * n_A + i * n_A + idx1];
    }
  }

  // Save distance.
  ScalarType dist = 0;
  for (int d = 0; d < DIM; ++d) {
    ScalarType diff =
        (B[d * n_B + idx2] -
         A[d * (LIFTING_DIM + 1) * n_A + LIFTING_DIM * n_A + idx1]);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      diff -= projection[i] * A[d * (LIFTING_DIM + 1) * n_A + i * n_A + idx1];
    }
    dist += diff * diff;
  }
  distances[idx1 * n_B + idx2] = dist;
}

template<int LIFTING_DIM> void compute_distances_between_affine_subspaces_and_points(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances) {
  assert(dim == DIM);
  assert(subspace_dim == LIFTING_DIM);

  // CUDA.
  cudaError_t cuda_status = cudaSuccess;

  // Allocate memory for distances.
  ScalarType* distances_device;
  cuda_status =
      cudaMalloc((void**)&distances_device, sizeof(ScalarType) * n_A * n_B);
  assert(cuda_status == cudaSuccess);

  // Run the kernel.
  const dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  const int n_blocks_A = (n_A + BLOCK_DIM - 1) / BLOCK_DIM;
  const int n_blocks_B = (n_B + BLOCK_DIM - 1) / BLOCK_DIM;
  const dim3 grid_dim(n_blocks_A, n_blocks_B);
  compute_distance_between_affine_subspace_and_point_kernel<LIFTING_DIM><<<grid_dim, block_dim>>>(
      n_A, A_device, n_B, B_device, distances_device);
  // Check for errors.
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(cuda_status) << std::endl;
    exit(1);
  }

  // Copy from device.
  cuda_status =
      cudaMemcpy(distances, distances_device, sizeof(ScalarType) * n_A * n_B,
                 cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);

  // Free memory.
  cudaFree(distances_device);
}

template void compute_distances_between_affine_subspaces_and_points<2>(
  const size_t dim, const size_t subspace_dim, const size_t n_A,
  const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
  ScalarType* distances);
template void compute_distances_between_affine_subspaces_and_points<4>(
  const size_t dim, const size_t subspace_dim, const size_t n_A,
  const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
  ScalarType* distances);
template void compute_distances_between_affine_subspaces_and_points<8>(
  const size_t dim, const size_t subspace_dim, const size_t n_A,
  const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
  ScalarType* distances);
template void compute_distances_between_affine_subspaces_and_points<16>(
  const size_t dim, const size_t subspace_dim, const size_t n_A,
  const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
  ScalarType* distances);

// Additional CUDA utilities.
ScalarType* copy_subspaces_to_device(const size_t dim, const size_t k_A,
                                     const size_t n_A, const ScalarType* A) {
  ScalarType* A_device;
  cudaError_t cuda_status =
      cudaMalloc((void**)&A_device, sizeof(ScalarType) * dim * (k_A + 1) * n_A);
  assert(cuda_status == cudaSuccess);
  cuda_status =
      cudaMemcpy(A_device, A, sizeof(ScalarType) * dim * (k_A + 1) * n_A,
                 cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
  return A_device;
}

ScalarType* copy_descriptors_to_device(const size_t dim, const size_t n_A,
                                       const ScalarType* A) {
  ScalarType* A_device;
  cudaError_t cuda_status =
      cudaMalloc((void**)&A_device, sizeof(ScalarType) * dim * n_A);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(A_device, A, sizeof(ScalarType) * dim * n_A,
                           cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
  return A_device;
}

void free_memory_from_device(ScalarType* A_device) {
  cudaFree(A_device);
}
