#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"
#include "matching.h"

template<int SUBSPACE_DIM> MatrixType subspace_to_subspace_exhaustive_matcher_t(const std::vector<Eigen::Ref<const MatrixType>>& subspaces1, const std::vector<Eigen::Ref<const MatrixType>>& subspaces2) {
  MatrixType squared_distances(subspaces1.size(), subspaces2.size());

  ScalarType* subspaces1_ptr = rearrange_subspaces_memory_layout(subspaces1);
  ScalarType* subspaces1_device_ptr = copy_subspaces_to_device(
    DIM, SUBSPACE_DIM, subspaces1.size(), subspaces1_ptr);

  ScalarType* subspaces2_ptr = rearrange_subspaces_memory_layout(subspaces2);
  ScalarType* subspaces2_device_ptr = copy_subspaces_to_device(
    DIM, SUBSPACE_DIM, subspaces2.size(), subspaces2_ptr);

  compute_distances_between_affine_subspaces<SUBSPACE_DIM>(
      DIM, SUBSPACE_DIM, subspaces1.size(), subspaces1_device_ptr, subspaces2.size(), subspaces2_device_ptr, squared_distances.data());

  free(subspaces1_ptr);
  free_memory_from_device(subspaces1_device_ptr);

  free(subspaces2_ptr);
  free_memory_from_device(subspaces2_device_ptr);

  return squared_distances;
}

MatrixType subspace_to_subspace_exhaustive_matcher(const std::vector<Eigen::Ref<const MatrixType>> subspaces1, const std::vector<Eigen::Ref<const MatrixType>> subspaces2) {
  if (subspaces1.size() == 0 || subspaces2.size() == 0) {
    MatrixType squared_distances(subspaces1.size(), subspaces2.size());
    return squared_distances;
  }
  const int subspace_dim = subspaces1[0].rows() - 1;
  if (subspace_dim == 2) {
    return subspace_to_subspace_exhaustive_matcher_t<2>(subspaces1, subspaces2);
  } else if (subspace_dim == 4) {
    return subspace_to_subspace_exhaustive_matcher_t<4>(subspaces1, subspaces2);
  } else {
    assert(false);
  }
}