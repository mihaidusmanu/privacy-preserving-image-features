#include <iostream>
#include <fstream>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"

ScalarType point_to_subspace_distance(const RowVectorType& p, const MatrixType& A) {
  assert(p.rows() == A.cols());

  const Eigen::Index subspace_dim = A.rows() - 1;

  const RowVectorType diff = p - A.row(subspace_dim);
  
  RowVectorType proj = A.row(subspace_dim);
  for (Eigen::Index i = 0; i < subspace_dim; ++i) {
    const RowVectorType u = A.row(i);
    proj += diff.dot(u) * u;
  }

  return (p - proj).squaredNorm();
}

MatrixType point_to_subspace_exhaustive_matcher(const std::vector<Eigen::Ref<const RowVectorType>> points, const std::vector<Eigen::Ref<const MatrixType>> subspaces) {
  MatrixType squared_distances(points.size(), subspaces.size());

  const int n_points = static_cast<int>(points.size());
  const int n_subspaces = static_cast<int>(subspaces.size());

#pragma omp parallel for
  for (int i = 0; i < n_points; ++i) {
    for (int j = 0; j < n_subspaces; ++j) {
      squared_distances(i, j) = point_to_subspace_distance(points[i], subspaces[j]);
    }
  }

  return squared_distances;
}
