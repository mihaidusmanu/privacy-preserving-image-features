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

// The closed-form solution below requires the system matrix J to be invertible.
// |det(J)| = det(N) with N = I - M M^T as in the paper, whose eigenvalues lie in
// [0, 1] (they are sin^2 of the principal angles between the subspaces), so
// det(N) <= lambda_min(N). If |det(J)| is above this tolerance, J is well
// conditioned and the closed form is accurate; otherwise the fallback is used.
// 1e-4 corresponds to a shared direction up to ~0.6 degrees. Same as the CUDA
// implementation.
constexpr ScalarType kSingularDetTolerance = 1e-4f;
// Pivots below this value are treated as zero by the fallback (directions closer
// than ~0.06 degrees are treated as shared). Below that, float32 cannot resolve the
// distance anyway: the closest points lie hundreds of units or more along the
// subspaces. Dropping a direction can only overestimate the distance.
constexpr ScalarType kSharedDirectionTolerance = 1e-6f;

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
    if (!(K(p, p) > kSharedDirectionTolerance)) {
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

// Fallback for (near-)singular J, i.e. subspaces that (almost) share a direction.
// This happens when two features are lifted with two or more of the same
// adversarial database words: both subspaces then contain the line through them.
// Solves the same equations as the closed form, eliminating beta as in the paper:
//   N alpha = yA + M yB,   beta = -M^T alpha - yB,   N = I - M M^T,  M = -bA bB^T,
// but with a rank-revealing solve instead of N^-1.
ScalarType subspace_to_subspace_distance_fallback(const MatrixType& A, const MatrixType& B) {
  const Eigen::Index subspace_dim_A = A.rows() - 1;
  const Eigen::Index subspace_dim_B = B.rows() - 1;

  const auto bA = A.topRows(subspace_dim_A);
  const auto bB = B.topRows(subspace_dim_B);
  const RowVectorType cA = A.row(subspace_dim_A);
  const RowVectorType cB = B.row(subspace_dim_B);
  const RowVectorType cAB = cB - cA;

  const MatrixType M = -bA * bB.transpose();
  const RowVectorType yA = cAB * bA.transpose();
  const RowVectorType yB = cAB * bB.transpose();
  MatrixType N = MatrixType::Identity(subspace_dim_A, subspace_dim_A) - M * M.transpose();
  const RowVectorType rhs = yA + yB * M.transpose();
  const RowVectorType coeff_A = solve_psd_system(N, rhs);
  const RowVectorType coeff_B = -coeff_A * M - yB;

  const RowVectorType pA = cA + coeff_A * bA;
  const RowVectorType pB = cB + coeff_B * bB;
  return (pB - pA).squaredNorm();
}

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

  // Solve linear system, falling back when J is (near-)singular; see
  // kSingularDetTolerance. The negated comparison also catches NaN.
  const Eigen::PartialPivLU<MatrixType> lu(J);
  if (!(std::abs(lu.determinant()) > kSingularDetTolerance)) {
    return subspace_to_subspace_distance_fallback(A, B);
  }
  const RowVectorType sol = lu.inverse() * y.transpose();

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
