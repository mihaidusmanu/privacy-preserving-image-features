#include "basic_types.h"

template<int LIFTING_DIM> extern void compute_distances_between_affine_subspaces(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances);

template<int LIFTING_DIM> extern void compute_distances_between_affine_subspaces_and_points(
    const size_t dim, const size_t subspace_dim, const size_t n_A,
    const ScalarType* A_device, const size_t n_B, const ScalarType* B_device,
    ScalarType* distances);

extern ScalarType* copy_subspaces_to_device(const size_t dim, const size_t k_A,
                                            const size_t n_A, const ScalarType* A);

extern ScalarType* copy_descriptors_to_device(const size_t dim, const size_t n_A,
                                              const ScalarType* A);

extern void free_memory_from_device(ScalarType* A_device);


#ifndef rearrange_layout
#define rearrange_layout

#include <vector>

#include <Eigen/Dense>

ScalarType* rearrange_subspaces_memory_layout(const std::vector<Eigen::Ref<const MatrixType>>& A) {
  const size_t n_A = A.size();
  if (n_A == 0) {
    return NULL;
  }
  const size_t dim = A[0].cols();
  const size_t k_A = A[0].rows() - 1;

  ScalarType* A_ptr = (ScalarType*)malloc(sizeof(ScalarType) * dim * (k_A + 1) * n_A);
  for (size_t i = 0; i < n_A; ++i) {
    for (size_t j = 0; j < k_A + 1; ++j) {
      for (size_t d = 0; d < dim; ++d) {
        A_ptr[d * (k_A + 1) * n_A + j * n_A + i] = A[i](j, d);
      }
    }
  }

  return A_ptr;
}

#endif
