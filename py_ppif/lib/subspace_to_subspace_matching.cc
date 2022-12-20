#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"

ScalarType subspace_to_subspace_distance(const MatrixType& A, const MatrixType& B) {
  assert(A.cols() == B.cols());

  const Eigen::Index subspace_dim_A = A.rows() - 1;
  const Eigen::Index subspace_dim_B = B.rows() - 1;

  const RowVectorType& cA = A.row(subspace_dim_A);
  const RowVectorType& cB = B.row(subspace_dim_B);
  const RowVectorType cAB = cB - cA;

  // Setup the righthand side vector y.
  RowVectorType y(subspace_dim_A + subspace_dim_B);
  for (Eigen::Index i = 0; i < subspace_dim_A; ++i) {
    y(i) = cAB.dot(A.row(i));
  }
  for (Eigen::Index i = 0; i < subspace_dim_B; ++i) {
    y(subspace_dim_A + i) = cAB.dot(B.row(i));
  }

  // Setup the square matrix J.
  MatrixType J(subspace_dim_A + subspace_dim_B, subspace_dim_A + subspace_dim_B);
  for (Eigen::Index i = 0; i < subspace_dim_A; ++i) {
    for (Eigen::Index j = 0; j < subspace_dim_A; ++j) {
      J(i, j) = 0;
    }
    J(i, i) = 1;
  }
  for (Eigen::Index i = 0; i < subspace_dim_A; ++i) {
    for (Eigen::Index j = 0; j < subspace_dim_B; ++j) {
      J(i, subspace_dim_A + j) = -A.row(i).dot(B.row(j));
    }
  }
  for (Eigen::Index i = 0; i < subspace_dim_B; ++i) {
    for (Eigen::Index j = 0; j < subspace_dim_A; ++j) {
      J(subspace_dim_A + i, j) = -J(j, subspace_dim_A + i);
    }
  }
  for (Eigen::Index i = 0; i < subspace_dim_B; ++i) {
    for (Eigen::Index j = 0; j < subspace_dim_B; ++j) {
      J(subspace_dim_A + i, subspace_dim_A + j) = 0;
    }
    J(subspace_dim_A + i, subspace_dim_A + i) = -1;
  }

  // Solve linear system.
  const RowVectorType sol = J.inverse() * y.transpose();

  // Compute the pair of neareast points.
  const RowVectorType& coeff_A = sol.segment(0, subspace_dim_A);
  const RowVectorType& coeff_B = sol.segment(subspace_dim_A, subspace_dim_B);

  RowVectorType pA = cA;
  for (Eigen::Index i = 0; i < subspace_dim_A; ++i) {
    pA += coeff_A(i) * A.row(i);
  }

  RowVectorType pB = cB;
  for (Eigen::Index i = 0; i < subspace_dim_B; ++i) {
    pB += coeff_B(i) * B.row(i);
  }

  // Compute and return their squared distance.
  return (pB - pA).squaredNorm();
}

MatrixType subspace_to_subspace_exhaustive_matcher(const std::vector<Eigen::Ref<const MatrixType>> subspaces1, const std::vector<Eigen::Ref<const MatrixType>> subspaces2) {
  MatrixType squared_distances(subspaces1.size(), subspaces2.size());

  const int n_subspaces1 = static_cast<int>(subspaces1.size());
  const int n_subspaces2 = static_cast<int>(subspaces2.size());

#pragma omp parallel for
  for (int i = 0; i < n_subspaces1; ++i) {
    for (int j = 0; j < n_subspaces2; ++j) {
      squared_distances(i, j) = subspace_to_subspace_distance(subspaces1[i], subspaces2[j]);
    }
  }

  return squared_distances;
}
