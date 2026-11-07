#include <iostream>
#include <fstream>
#include <assert.h>

#include <cuda_runtime.h>

#include "basic_types.h"
#include "matrix_utils.cu"

#define BLOCK_DIM 16

// The closed-form solution below inverts N = I - M M^T and requires it to be
// invertible. The eigenvalues of N lie in [0, 1] (they are sin^2 of the principal
// angles between the subspaces), so det(N) <= lambda_min(N). If det(N) is above
// this tolerance, N is well conditioned and the closed form is accurate; otherwise
// the fallback below is used. 1e-4 corresponds to a shared direction up to ~0.6
// degrees, below which the closed form starts losing accuracy in float.
#define SINGULAR_DET_TOLERANCE 1e-4f
// Pivots below this value are treated as zero by the fallback (directions closer
// than ~0.06 degrees are treated as shared). Below that, float32 cannot resolve the
// distance anyway: the closest points lie hundreds of units or more along the
// subspaces. Dropping a direction can only overestimate the distance.
#define SHARED_DIRECTION_TOLERANCE 1e-6f

// Fallback for (near-)singular N, i.e. subspaces that (almost) share a direction.
// This happens when two features are lifted with two or more of the same
// adversarial database words: both subspaces then contain the line through them.
// Solves the same equations as the closed form in the kernel below,
//   N alpha = yA + M yB,   beta = -M^T alpha - yB,
// but with a rank-revealing solve instead of N^-1. It recomputes its inputs from
// global memory and is kept out of line, so that it does not affect register
// allocation of the common path.
template<int LIFTING_DIM> __device__ __noinline__ ScalarType distance_between_two_affine_subspaces_fallback(
    const int n_A, const ScalarType* A, const int n_B, const ScalarType* B,
    const int idx1, const int idx2) {
// Element d of row i of subspace idx1 (A) / idx2 (B); row LIFTING_DIM is the offset.
#define A_AT(i, d) A[(d) * (LIFTING_DIM + 1) * n_A + (i) * n_A + idx1]
#define B_AT(i, d) B[(d) * (LIFTING_DIM + 1) * n_B + (i) * n_B + idx2]
  ScalarType yA[LIFTING_DIM];
  ScalarType yB[LIFTING_DIM];
  ScalarType M[LIFTING_DIM * LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    yA[i] = 0;
    yB[i] = 0;
    for (int j = 0; j < LIFTING_DIM; ++j) {
      M[i * LIFTING_DIM + j] = 0;
    }
  }
  for (int d = 0; d < DIM; ++d) {
    const ScalarType cAB = B_AT(LIFTING_DIM, d) - A_AT(LIFTING_DIM, d);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      yA[i] += A_AT(i, d) * cAB;
      yB[i] += B_AT(i, d) * cAB;
      for (int j = 0; j < LIFTING_DIM; ++j) {
        M[i * LIFTING_DIM + j] -= A_AT(i, d) * B_AT(j, d);
      }
    }
  }

  // N = I - M M^T and the right-hand side yA + M yB.
  ScalarType N[LIFTING_DIM * LIFTING_DIM];
  ScalarType rhs[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    rhs[i] = yA[i];
    for (int j = 0; j < LIFTING_DIM; ++j) {
      rhs[i] += M[i * LIFTING_DIM + j] * yB[j];
      ScalarType v = (i == j) ? 1.0f : 0.0f;
      for (int k = 0; k < LIFTING_DIM; ++k) {
        v -= M[i * LIFTING_DIM + k] * M[j * LIFTING_DIM + k];
      }
      N[i * LIFTING_DIM + j] = v;
    }
  }

  ScalarType xA[LIFTING_DIM];
  solve_psd_system<LIFTING_DIM>(N, rhs, xA, SHARED_DIRECTION_TOLERANCE);
  ScalarType xB[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    xB[i] = -yB[i];
    for (int j = 0; j < LIFTING_DIM; ++j) {
      xB[i] -= M[j * LIFTING_DIM + i] * xA[j];
    }
  }

  ScalarType dist = 0;
  for (int d = 0; d < DIM; ++d) {
    ScalarType aux = B_AT(LIFTING_DIM, d) - A_AT(LIFTING_DIM, d);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      aux += xB[i] * B_AT(i, d) - xA[i] * A_AT(i, d);
    }
    dist += aux * aux;
  }
  return dist;
#undef A_AT
#undef B_AT
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

  // J = [ bA * bA^T, -bA * bB^T ]
  //     [ bB * bA^T, -bB * bB^T ].
  // y = [ bA * cAB ]
  //     [ bB * cAB ].
  // We notice that the J matrix has the following form:
  // J = [  I  ,  M ]
  //     [ -M^T, -I ].
  // We can invert J using the block matrix inversion formula.
  // Let N = (I - M M^T)^-1.
  // The inverse can be computed as follows:
  // [   N          N M     ]
  // [ -M^T N  -M^T N M - I ].

  // Set up the Kx1 right-side vector y.
  ScalarType yA[LIFTING_DIM];
  ScalarType yB[LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    yA[i] = 0;
    yB[i] = 0;
    for (int d = 0; d < DIM; ++d) {
      const ScalarType cAB =
          (B[d * (LIFTING_DIM + 1) * n_B + LIFTING_DIM * n_B + idx2] -
           A[d * (LIFTING_DIM + 1) * n_A + LIFTING_DIM * n_A + idx1]);
      yA[i] += A[d * (LIFTING_DIM + 1) * n_A + i * n_A + idx1] * cAB;
      yB[i] += B[d * (LIFTING_DIM + 1) * n_B + i * n_B + idx2] * cAB;
    }
  }

  // Set up the matrix M.
  ScalarType M[LIFTING_DIM * LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    for (int j = 0; j < LIFTING_DIM; ++j) {
      const int idx = i * LIFTING_DIM + j;
      M[idx] = 0;
      for (int d = 0; d < DIM; ++d) {
        M[idx] -= A[d * (LIFTING_DIM + 1) * n_A + i * n_A + idx1] *
                  B[d * (LIFTING_DIM + 1) * n_B + j * n_B + idx2];
      }
    }
  }

  // Compute (I - M M^T).
  ScalarType N[LIFTING_DIM * LIFTING_DIM];
  for (int i = 0; i < LIFTING_DIM; ++i) {
    for (int j = 0; j < LIFTING_DIM; ++j) {
      const int idx = i * LIFTING_DIM + j;
      N[idx] = 0;
      for (int k = 0; k < LIFTING_DIM; ++k) {
        N[idx] += (-1.0) * M[i * LIFTING_DIM + k] * M[j * LIFTING_DIM + k];
      }
    }
    N[i * LIFTING_DIM + i] += 1;
  }

  // Fall back when N is (near-)singular; see SINGULAR_DET_TOLERANCE. The
  // negated comparison also catches NaN.
  if (!(determinant<LIFTING_DIM>(N) > SINGULAR_DET_TOLERANCE)) {
    distances[idx1 * n_B + idx2] =
        distance_between_two_affine_subspaces_fallback<LIFTING_DIM>(n_A, A, n_B, B, idx1, idx2);
    return;
  }

  // Compute N = (I - M M^T)^-1.
  invert_matrix_in_place<LIFTING_DIM>(N);

  // Compute solution.
  // Define solution for A.
  ScalarType xA[LIFTING_DIM];
  // Top left block: N.
  for (int i = 0; i < LIFTING_DIM; ++i) {
    xA[i] = 0;
    for (int j = 0; j < LIFTING_DIM; ++j) {
      xA[i] += N[i * LIFTING_DIM + j] * yA[j];
    }
  }
  // Top right block: N M.
  for (int i = 0; i < LIFTING_DIM; ++i) {
    for (int j = 0; j < LIFTING_DIM; ++j) {
      ScalarType aux = 0;
      for (int k = 0; k < LIFTING_DIM; ++k) {
        aux += N[i * LIFTING_DIM + k] * M[k * LIFTING_DIM + j];
      }
      xA[i] += aux * yB[j];
    }
  }
  // Define solution for B.
  ScalarType xB[LIFTING_DIM];
  // Bottom left block: -M^T N.
  // Overwrite N by M^T N.
  left_transpose_multiply_matrix_in_place<LIFTING_DIM>(N, M);
  for (int i = 0; i < LIFTING_DIM; ++i) {
    xB[i] = 0;
    for (int j = 0; j < LIFTING_DIM; ++j) {
      xB[i] += (-N[i * LIFTING_DIM + j]) * yA[j];
    }
  }
  // Bottom right block: -M^T N M - I = -(M^T N M + I).
  // Overwrite N = M^T N by M^T N M.
  right_multiply_matrix_in_place<LIFTING_DIM>(N, M);
  for (int i = 0; i < LIFTING_DIM; ++i) {
    // Add I.
    N[i * LIFTING_DIM + i] += 1;
    for (int j = 0; j < LIFTING_DIM; ++j) {
      xB[i] += (-N[i * LIFTING_DIM + j]) * yB[j];
    }
  }

  // Save distance.
  ScalarType dist = 0;
  for (int d = 0; d < DIM; ++d) {
    ScalarType aux =
      (B[d * (LIFTING_DIM + 1) * n_B + LIFTING_DIM * n_B + idx2] -
       A[d * (LIFTING_DIM + 1) * n_A + LIFTING_DIM * n_A + idx1]);
    for (int i = 0; i < LIFTING_DIM; ++i) {
      aux += (xB[i] * B[d * (LIFTING_DIM + 1) * n_B + i * n_B + idx2] -
              xA[i] * A[d * (LIFTING_DIM + 1) * n_A + i * n_A + idx1]);
    }
    dist += aux * aux;
  }
  distances[idx1 * n_B + idx2] = dist;
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
