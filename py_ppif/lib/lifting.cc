#include <iostream>
#include <fstream>
#include <assert.h>
#include <cstdint>
#include <random>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

#include "types.h"

// Random number generator used by the lifting functions, created per call from
// the given seed.
//
// This replaces std::srand / std::rand (also used internally by Eigen's
// Random()), which share process-wide state and are implementation-defined:
// on macOS, the first std::rand() after std::srand(s) is 16807 * s, so
// consecutive seeds (e.g. consecutive image ids) gave strongly correlated
// sub-database choices, and on Windows RAND_MAX is only 32767, so most of a
// large lifting database could never be sampled. std::mt19937 and
// std::seed_seq are fully specified by the standard, and the mappings below
// avoid std::uniform_*_distribution (whose output differs between standard
// libraries), so lifting gives the same result on every platform.
class LiftingRng {
 public:
  explicit LiftingRng(const int seed) {
    std::seed_seq seed_sequence{seed};
    engine_.seed(seed_sequence);
  }

  // Uniform integer in [0, n), using rejection sampling to avoid modulo bias.
  uint32_t uniform_index(const uint32_t n) {
    assert(n > 0);
    const uint64_t range = uint64_t(1) << 32;
    const uint64_t limit = range - range % n;
    uint64_t x;
    do {
      x = static_cast<uint32_t>(engine_());
    } while (x >= limit);
    return static_cast<uint32_t>(x % n);
  }

  // Vector with entries drawn uniformly from [-1, 1], like Eigen's Random().
  RowVectorType random_vector(const Eigen::Index n) {
    RowVectorType v(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const double u = static_cast<uint32_t>(engine_()) / 4294967295.0;
      v(i) = static_cast<ScalarType>(2.0 * u - 1.0);
    }
    return v;
  }

 private:
  std::mt19937 engine_;
};

RowVectorType orthonormal_projection(const RowVectorType& u, const RowVectorType& v) {
  return (u.dot(v) / u.dot(u)) * u;
}

MatrixType gram_schmidt_orthogonalization(const MatrixType& B) {
  const Eigen::Index subspace_dim = B.rows();
  const Eigen::Index dim = B.cols();
  MatrixType M(subspace_dim, dim);
  M.row(0) = B.row(0).normalized();

  for (Eigen::Index i = 1; i < subspace_dim; ++i) {
    RowVectorType v = B.row(i);
    for (Eigen::Index j = 0; j < i; ++j) {
      const RowVectorType& u = M.row(j);
      v = v - orthonormal_projection(u, v);
    }
    M.row(i) = v.normalized();
  }

  return M;
}

RowVectorType project_point_to_subspace(const RowVectorType& p, const MatrixType& M) {
  const Eigen::Index subspace_dim = M.rows() - 1;
  RowVectorType proj = M.row(subspace_dim);
  const RowVectorType pt = p - proj;
  for (Eigen::Index i = 0; i < subspace_dim; ++i) {
    proj += pt.dot(M.row(i)) * M.row(i);
  }
  return proj;
}

MatrixType anonymize_subspace(const MatrixType& M, LiftingRng& rng) {
  const Eigen::Index subspace_dim = M.rows() - 1;
  MatrixType private_M(subspace_dim + 1, DIM);

  // Anonymize the offset.
  private_M.row(subspace_dim) = project_point_to_subspace(rng.random_vector(DIM), M);

  // Anonymize the basis.
  for (Eigen::Index i = 0; i < subspace_dim; ++i) {
    private_M.row(i) = project_point_to_subspace(rng.random_vector(DIM), M) - private_M.row(subspace_dim);
  }

  // Orthogonalize final basis.
  const MatrixType& top = private_M.topRows(subspace_dim);
  private_M.topRows(subspace_dim) = gram_schmidt_orthogonalization(top);

  return private_M;
}

std::vector<MatrixType> random_lifting(
    const Eigen::Ref<const MatrixType>& descriptors, const int subspace_dim,
    const int seed) {
  LiftingRng rng(seed);

  assert(descriptors.cols() == DIM);
  const int num = static_cast<int>(descriptors.rows());

  std::vector<MatrixType> A(num);

  for (int i = 0; i < num; ++i) {
    MatrixType M(subspace_dim + 1, DIM);

    // Generate subspace_dim random directions.
    for (Eigen::Index j = 0; j < subspace_dim; ++j) {
      M.row(j) = rng.random_vector(DIM);
    }

    // Orthogonalize basis.
    const MatrixType& top = M.topRows(subspace_dim);
    M.topRows(subspace_dim) = gram_schmidt_orthogonalization(top);

    // Set subspace offset to original descriptor.
    M.row(subspace_dim) = descriptors.row(i);

    // Anonymize the subspace.
    A[i] = anonymize_subspace(M, rng);
  }

  return A;
}


std::vector<MatrixType> adversarial_lifting(
    const Eigen::Ref<const MatrixType>& descriptors, const int subspace_dim,
    const Eigen::Ref<const MatrixType>& database, const int num_sub_databases,
    const int seed) {
  LiftingRng rng(seed);

  assert(descriptors.cols() == DIM);
  const int num = static_cast<int>(descriptors.rows());

  assert(database.rows() % num_sub_databases == 0);
  const int sub_database_size = database.rows() / num_sub_databases;
  const int sub_database_idx = rng.uniform_index(num_sub_databases);

  std::vector<MatrixType> A(num);

  for (int i = 0; i < num; ++i) {
    MatrixType M(subspace_dim + 1, DIM);

    // Generate subspace_dim adversarial directions.
    std::set<size_t> random_word_indices;
    for (Eigen::Index j = 0; j < subspace_dim; ++j) {
      // Pick a random index (without repetition).
      size_t random_word_index = -1;
      while (true) {
        random_word_index = rng.uniform_index(sub_database_size);
        if (random_word_indices.count(random_word_index) == 0) {
          random_word_indices.insert(random_word_index);
          break;
        }
      }

      M.row(j) = database.row(sub_database_idx * sub_database_size + random_word_index) - descriptors.row(i);
    }
    
    // Orthogonalize basis.
    const MatrixType& top = M.topRows(subspace_dim);
    M.topRows(subspace_dim) = gram_schmidt_orthogonalization(top);

    // Set subspace offset to original descriptor.
    M.row(subspace_dim) = descriptors.row(i);

    // Anonymize the subspace.
    A[i] = anonymize_subspace(M, rng);
  }

  return A;
}

std::vector<MatrixType> hybrid_lifting(
    const Eigen::Ref<const MatrixType>& descriptors, const int subspace_dim,
    const Eigen::Ref<const MatrixType>& database, const int num_sub_databases,
    const int seed) {
  LiftingRng rng(seed);

  assert(descriptors.cols() == DIM);
  const int num = static_cast<int>(descriptors.rows());

  assert(database.rows() % num_sub_databases == 0);
  const int sub_database_size = database.rows() / num_sub_databases;
  const int sub_database_idx = rng.uniform_index(num_sub_databases);

  std::vector<MatrixType> A(num);

  for (int i = 0; i < num; ++i) {
    MatrixType M(subspace_dim + 1, DIM);

    // Generate subspace_dim / 2 adversarial directions.
    std::set<size_t> random_word_indices;
    for (Eigen::Index j = 0; j < subspace_dim / 2; ++j) {
      // Pick a random index (without repetition).
      size_t random_word_index = -1;
      while (true) {
        random_word_index = rng.uniform_index(sub_database_size);
        if (random_word_indices.count(random_word_index) == 0) {
          random_word_indices.insert(random_word_index);
          break;
        }
      }

      M.row(j) = database.row(sub_database_idx * sub_database_size + random_word_index) - descriptors.row(i);
    }
    // Generate subspace_dim / 2 random directions.
    for (Eigen::Index j = subspace_dim / 2; j < subspace_dim; ++j) {
      M.row(j) = rng.random_vector(DIM);
    }
    
    // Orthogonalize basis.
    const MatrixType& top = M.topRows(subspace_dim);
    M.topRows(subspace_dim) = gram_schmidt_orthogonalization(top);

    // Set subspace offset to original descriptor.
    M.row(subspace_dim) = descriptors.row(i);

    // Anonymize the subspace.
    A[i] = anonymize_subspace(M, rng);
  }

  return A;
}
