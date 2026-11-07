#include <iostream>
#include <fstream>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"
#include "matching.h"

template<int SUBSPACE_DIM> MatrixType point_to_subspace_exhaustive_matcher_t(const Eigen::Ref<const MatrixType> points, const std::vector<Eigen::Ref<const MatrixType>>& subspaces) {
  const size_t n_points = points.rows();
  const size_t n_subspaces = subspaces.size();

  const MatrixType points_transpose = points.transpose();
  ScalarType* points_device_ptr = copy_descriptors_to_device(
      DIM, n_points, points_transpose.data());

  ScalarType* subspaces_ptr = rearrange_subspaces_memory_layout(subspaces);
  ScalarType* subspaces_device_ptr = copy_subspaces_to_device(
    DIM, SUBSPACE_DIM, n_subspaces, subspaces_ptr);

  // The kernel takes the subspaces as its first operand (A) and the points as
  // its second (B), and writes a row-major n_subspaces x n_points matrix.
  MatrixType squared_distances_transpose(n_subspaces, n_points);
  compute_distances_between_affine_subspaces_and_points<SUBSPACE_DIM>(
      DIM, SUBSPACE_DIM, n_subspaces, subspaces_device_ptr, n_points, points_device_ptr, squared_distances_transpose.data());

  free_memory_from_device(points_device_ptr);

  free(subspaces_ptr);
  free_memory_from_device(subspaces_device_ptr);

  // Return n_points x n_subspaces, matching the CPU implementation.
  return squared_distances_transpose.transpose();
}

MatrixType point_to_subspace_exhaustive_matcher(const Eigen::Ref<const MatrixType> points, const std::vector<Eigen::Ref<const MatrixType>> subspaces) {
  if (points.rows() == 0 || subspaces.size() == 0) {
    MatrixType squared_distances(points.rows(), subspaces.size());
    return squared_distances;
  }
  const int subspace_dim = subspaces[0].rows() - 1;
  if (subspace_dim == 2) {
    return point_to_subspace_exhaustive_matcher_t<2>(points, subspaces);
  } else if (subspace_dim == 4) {
    return point_to_subspace_exhaustive_matcher_t<4>(points, subspaces);
  } else if (subspace_dim == 8) {
    return point_to_subspace_exhaustive_matcher_t<8>(points, subspaces);
  } else if (subspace_dim == 16) {
    return point_to_subspace_exhaustive_matcher_t<16>(points, subspaces);
  } else {
    assert(false);
  }
}
