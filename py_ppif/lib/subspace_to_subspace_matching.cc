#include <cmath>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"

// Pivots of I - C^T C below this value are treated as zero. The eigenvalues of
// I - C^T C are sin^2 of the principal angles between the two subspaces, so this
// treats directions closer than ~0.2 degrees as shared.
constexpr ScalarType kSharedDirectionTolerance = 1e-5f;

// Solves the symmetric positive semi-definite system K x = z (with z in the range
// of K) using a pivoted Cholesky factorisation K = P L L^T P^T. Pivots below
// kSharedDirectionTolerance are treated as zero and the matching entries of x are
// set to 0, so a singular K gives a valid solution instead of dividing by ~0.
// K is overwritten: on return, its lower triangle holds L.
RowVectorType solve_psd_system(MatrixType& K, const RowVectorType& z) {
  const Eigen::Index n = K.rows();
  std::vector<Eigen::Index> perm(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    perm[i] = i;
  }

  Eigen::Index rank = 0;
  for (Eigen::Index k = 0; k < n; ++k) {
    // Pivot on the largest remaining diagonal entry.
    Eigen::Index p = k;
    for (Eigen::Index i = k + 1; i < n; ++i) {
      if (K(i, i) > K(p, p)) {
        p = i;
      }
    }
    if (K(p, p) <= kSharedDirectionTolerance) {
      break;
    }
    if (p != k) {
      K.row(k).swap(K.row(p));
      K.col(k).swap(K.col(p));
      std::swap(perm[k], perm[p]);
    }
    // Column k of L, then update the trailing block (kept symmetric so that
    // later row / column swaps stay valid).
    K(k, k) = std::sqrt(K(k, k));
    for (Eigen::Index i = k + 1; i < n; ++i) {
      K(i, k) /= K(k, k);
    }
    for (Eigen::Index i = k + 1; i < n; ++i) {
      for (Eigen::Index j = k + 1; j < n; ++j) {
        K(i, j) -= K(i, k) * K(j, k);
      }
    }
    ++rank;
  }

  // Solve L L^T w = P^T z on the leading rank x rank block.
  RowVectorType w(rank);
  for (Eigen::Index i = 0; i < rank; ++i) {
    ScalarType s = z(perm[i]);
    for (Eigen::Index j = 0; j < i; ++j) {
      s -= K(i, j) * w(j);
    }
    w(i) = s / K(i, i);
  }
  for (Eigen::Index i = rank - 1; i >= 0; --i) {
    ScalarType s = w(i);
    for (Eigen::Index j = i + 1; j < rank; ++j) {
      s -= K(j, i) * w(j);
    }
    w(i) = s / K(i, i);
  }

  RowVectorType x = RowVectorType::Zero(n);
  for (Eigen::Index i = 0; i < rank; ++i) {
    x(perm[i]) = w(i);
  }
  return x;
}

ScalarType subspace_to_subspace_distance(const MatrixType& A, const MatrixType& B) {
  assert(A.cols() == B.cols());

  const Eigen::Index subspace_dim_A = A.rows() - 1;
  const Eigen::Index subspace_dim_B = B.rows() - 1;

  // Orthonormal bases (D and E in the paper) and offsets.
  const auto bA = A.topRows(subspace_dim_A);
  const auto bB = B.topRows(subspace_dim_B);
  const RowVectorType cA = A.row(subspace_dim_A);
  const RowVectorType cB = B.row(subspace_dim_B);
  const RowVectorType cAB = cB - cA;

  // We look for pA = cA + alpha^T bA and pB = cB + beta^T bB minimising
  // ||pB - pA||. For a fixed beta, the optimal alpha is
  //   alpha = bA (cAB + bB^T beta) = yA + C beta,
  // with C = bA bB^T, yA = bA cAB and yB = bB cAB. Substituting it back leaves
  //   (I - C^T C) beta = -(yB - C^T yA),
  // which is the paper's 2m x 2m system with alpha eliminated. I - C^T C is
  // singular exactly when the subspaces share a direction (e.g. two features
  // lifted with the same two adversarial database words), where the paper's
  // block inversion of N = I - M M^T divides by ~0. Solving it as a
  // semi-definite system keeps the distance correct in that case.
  const MatrixType C = bA * bB.transpose();
  const RowVectorType yA = cAB * bA.transpose();
  const RowVectorType yB = cAB * bB.transpose();
  MatrixType K = MatrixType::Identity(subspace_dim_B, subspace_dim_B) - C.transpose() * C;
  const RowVectorType z = yB - yA * C;
  const RowVectorType beta = -solve_psd_system(K, z);
  const RowVectorType alpha = yA + beta * C.transpose();

  // Compute the pair of nearest points and return their squared distance.
  const RowVectorType pA = cA + alpha * bA;
  const RowVectorType pB = cB + beta * bB;
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
