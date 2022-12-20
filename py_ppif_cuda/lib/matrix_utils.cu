#include <iostream>
#include <fstream>
#include <assert.h>

#include <cuda_runtime.h>

#include "basic_types.h"

#define determinant_matrix2(m00, m01, \
                            m10, m11) \
  (m00 * m11 - m01 * m10)
#define determinant_matrix3(m00, m01, m02, \
                            m10, m11, m12, \
                            m20, m21, m22) \
  (m00 * determinant_matrix2(m11, m12, m21, m22) - \
   m01 * determinant_matrix2(m10, m12, m20, m22) + \
   m02 * determinant_matrix2(m10, m11, m20, m21))
#define determinant_matrix4(m00, m01, m02, m03, \
                            m10, m11, m12, m13, \
                            m20, m21, m22, m23, \
                            m30, m31, m32, m33) \
  (m00 * determinant_matrix3(m11, m12, m13, m21, m22, m23, m31, m32, m33) - \
   m01 * determinant_matrix3(m10, m12, m13, m20, m22, m23, m30, m32, m33) + \
   m02 * determinant_matrix3(m10, m11, m13, m20, m21, m23, m30, m31, m33) - \
   m03 * determinant_matrix3(m10, m11, m12, m20, m21, m22, m30, m31, m32))

/// Invert a NxN matrix in place.
/// M <- M^(-1)
///
/// @param M  NxN row-major matrix.
template<int N> __device__ void invert_matrix_in_place(ScalarType* M);

template<> __device__ void invert_matrix_in_place<2>(ScalarType* M) {
  // [ 0, 1 ] [ a, b ]
  // [ 2, 3 ] [ c, d ]
  const ScalarType a = M[0];
  const ScalarType b = M[1];
  const ScalarType c = M[2];
  const ScalarType d = M[3];
  const ScalarType inv_det = 1.0 / determinant_matrix2(a, b, c, d);
  M[0] = inv_det * d;
  M[1] = (-1.0) * inv_det * b;
  M[2] = (-1.0) * inv_det * c;
  M[3] = inv_det * a;
}

template<> __device__ void invert_matrix_in_place<4>(ScalarType* M) {
  const ScalarType m00 = M[0];
  const ScalarType m01 = M[1];
  const ScalarType m02 = M[2];
  const ScalarType m03 = M[3];
  const ScalarType m10 = M[4];
  const ScalarType m11 = M[5];
  const ScalarType m12 = M[6];
  const ScalarType m13 = M[7];
  const ScalarType m20 = M[8];
  const ScalarType m21 = M[9];
  const ScalarType m22 = M[10];
  const ScalarType m23 = M[11];
  const ScalarType m30 = M[12];
  const ScalarType m31 = M[13];
  const ScalarType m32 = M[14];
  const ScalarType m33 = M[15];
  const ScalarType inv_det =
      1.0 / determinant_matrix4(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21,
                           m22, m23, m30, m31, m32, m33);
  // M^-1 = 1 / det(M) * C^T
  // C_{i, j} = det(M_{i, j})
  // M_{i, j} = M without row i and column j
  // First row.
  M[0] = inv_det *
         determinant_matrix3(m11, m12, m13, m21, m22, m23, m31, m32, m33);  // 0, 0
  M[4] = inv_det * (-1) *
         determinant_matrix3(m10, m12, m13, m20, m22, m23, m30, m32, m33);  // 0, 1
  M[8] = inv_det *
         determinant_matrix3(m10, m11, m13, m20, m21, m23, m30, m31, m33);  // 0, 2
  M[12] = inv_det * (-1) *
         determinant_matrix3(m10, m11, m12, m20, m21, m22, m30, m31, m32);  // 0, 3
  // Second row.
  M[1] = inv_det * (-1) *
         determinant_matrix3(m01, m02, m03, m21, m22, m23, m31, m32, m33);  // 1, 0
  M[5] = inv_det *
         determinant_matrix3(m00, m02, m03, m20, m22, m23, m30, m32, m33);  // 1, 1
  M[9] = inv_det * (-1) *
         determinant_matrix3(m00, m01, m03, m20, m21, m23, m30, m31, m33);  // 1, 2
  M[13] = inv_det *
         determinant_matrix3(m00, m01, m02, m20, m21, m22, m30, m31, m32);  // 1, 3
  // Third row.
  M[2] = inv_det *
         determinant_matrix3(m01, m02, m03, m11, m12, m13, m31, m32, m33);  // 2, 0
  M[6] = inv_det * (-1) *
         determinant_matrix3(m00, m02, m03, m10, m12, m13, m30, m32, m33);  // 2, 1
  M[10] = inv_det *
          determinant_matrix3(m00, m01, m03, m10, m11, m13, m30, m31, m33);  // 2, 2
  M[14] = inv_det * (-1) *
          determinant_matrix3(m00, m01, m02, m10, m11, m12, m30, m31, m32);  // 2, 3
  // Fourth row.
  M[3] = inv_det * (-1) *
         determinant_matrix3(m01, m02, m03, m11, m12, m13, m21, m22, m23);  // 3, 0
  M[7] = inv_det *
         determinant_matrix3(m00, m02, m03, m10, m12, m13, m20, m22, m23);  // 3, 1
  M[11] = inv_det * (-1) *
          determinant_matrix3(m00, m01, m03, m10, m11, m13, m20, m21, m23);  // 3, 2
  M[15] = inv_det *
          determinant_matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22);  // 3, 3
}

/// Right matrix multiplication in place.
/// M <- M * M_
///
/// @param M  NxN row-major matrix.
/// @param M_ NxN row-major matrix.
template<int N> __device__ void right_multiply_matrix_in_place(ScalarType* M, const ScalarType* M_);

template<> __device__ void right_multiply_matrix_in_place<2>(ScalarType* M, const ScalarType* M_) {
  // [ 0, 1 ] [ a, b ]
  // [ 2, 3 ] [ c, d ]
  const ScalarType a = M[0];
  const ScalarType b = M[1];
  const ScalarType c = M[2];
  const ScalarType d = M[3];
  M[0] = a * M_[0] + b * M_[2];
  M[1] = a * M_[1] + b * M_[3];
  M[2] = c * M_[0] + d * M_[2];
  M[3] = c * M_[1] + d * M_[3];
}

template<> __device__ void right_multiply_matrix_in_place<4>(ScalarType* M, const ScalarType* M_) {
  const ScalarType m00 = M[0];
  const ScalarType m01 = M[1];
  const ScalarType m02 = M[2];
  const ScalarType m03 = M[3];
  const ScalarType m10 = M[4];
  const ScalarType m11 = M[5];
  const ScalarType m12 = M[6];
  const ScalarType m13 = M[7];
  const ScalarType m20 = M[8];
  const ScalarType m21 = M[9];
  const ScalarType m22 = M[10];
  const ScalarType m23 = M[11];
  const ScalarType m30 = M[12];
  const ScalarType m31 = M[13];
  const ScalarType m32 = M[14];
  const ScalarType m33 = M[15];
  // First row.
  M[0] = m00 * M_[0] + m01 * M_[4] + m02 * M_[8] + m03 * M_[12];
  M[1] = m00 * M_[1] + m01 * M_[5] + m02 * M_[9] + m03 * M_[13];
  M[2] = m00 * M_[2] + m01 * M_[6] + m02 * M_[10] + m03 * M_[14];
  M[3] = m00 * M_[3] + m01 * M_[7] + m02 * M_[11] + m03 * M_[15];
  // Second row.
  M[4] = m10 * M_[0] + m11 * M_[4] + m12 * M_[8] + m13 * M_[12];
  M[5] = m10 * M_[1] + m11 * M_[5] + m12 * M_[9] + m13 * M_[13];
  M[6] = m10 * M_[2] + m11 * M_[6] + m12 * M_[10] + m13 * M_[14];
  M[7] = m10 * M_[3] + m11 * M_[7] + m12 * M_[11] + m13 * M_[15];
  // Third row.
  M[8] = m20 * M_[0] + m21 * M_[4] + m22 * M_[8] + m23 * M_[12];
  M[9] = m20 * M_[1] + m21 * M_[5] + m22 * M_[9] + m23 * M_[13];
  M[10] = m20 * M_[2] + m21 * M_[6] + m22 * M_[10] + m23 * M_[14];
  M[11] = m20 * M_[3] + m21 * M_[7] + m22 * M_[11] + m23 * M_[15];
  // Fourth row.
  M[12] = m30 * M_[0] + m31 * M_[4] + m32 * M_[8] + m33 * M_[12];
  M[13] = m30 * M_[1] + m31 * M_[5] + m32 * M_[9] + m33 * M_[13];
  M[14] = m30 * M_[2] + m31 * M_[6] + m32 * M_[10] + m33 * M_[14];
  M[15] = m30 * M_[3] + m31 * M_[7] + m32 * M_[11] + m33 * M_[15];
}

/// Left transpose matrix multiplication in place.
/// M <- M_^T * M
///
/// @param M  NxN row-major matrix.
/// @param M_ NxN row-major matrix.
template<int N> __device__ void left_transpose_multiply_matrix_in_place(ScalarType* M, const ScalarType* M_);

template<> __device__ void left_transpose_multiply_matrix_in_place<2>(ScalarType* M, const ScalarType* M_) {
  // [ 0, 1 ] [ a, b ]
  // [ 2, 3 ] [ c, d ]
  const ScalarType a = M[0];
  const ScalarType b = M[1];
  const ScalarType c = M[2];
  const ScalarType d = M[3];
  M[0] = M_[0] * a + M_[2] * c;
  M[1] = M_[0] * b + M_[2] * d;
  M[2] = M_[1] * a + M_[3] * c;
  M[3] = M_[1] * b + M_[3] * d;
}

template<> __device__ void left_transpose_multiply_matrix_in_place<4>(ScalarType* M, const ScalarType* M_) {
  const ScalarType m00 = M[0];
  const ScalarType m01 = M[1];
  const ScalarType m02 = M[2];
  const ScalarType m03 = M[3];
  const ScalarType m10 = M[4];
  const ScalarType m11 = M[5];
  const ScalarType m12 = M[6];
  const ScalarType m13 = M[7];
  const ScalarType m20 = M[8];
  const ScalarType m21 = M[9];
  const ScalarType m22 = M[10];
  const ScalarType m23 = M[11];
  const ScalarType m30 = M[12];
  const ScalarType m31 = M[13];
  const ScalarType m32 = M[14];
  const ScalarType m33 = M[15];
  // First row.
  M[0] = M_[0] * m00 + M_[4] * m10 + M_[8] * m20 + M_[12] * m30;
  M[1] = M_[0] * m01 + M_[4] * m11 + M_[8] * m21 + M_[12] * m31;
  M[2] = M_[0] * m02 + M_[4] * m12 + M_[8] * m22 + M_[12] * m32;
  M[3] = M_[0] * m03 + M_[4] * m13 + M_[8] * m23 + M_[12] * m33;
  // Second row.
  M[4] = M_[1] * m00 + M_[5] * m10 + M_[9] * m20 + M_[13] * m30;
  M[5] = M_[1] * m01 + M_[5] * m11 + M_[9] * m21 + M_[13] * m31;
  M[6] = M_[1] * m02 + M_[5] * m12 + M_[9] * m22 + M_[13] * m32;
  M[7] = M_[1] * m03 + M_[5] * m13 + M_[9] * m23 + M_[13] * m33;
  // Third row.
  M[8] = M_[2] * m00 + M_[6] * m10 + M_[10] * m20 + M_[14] * m30;
  M[9] = M_[2] * m01 + M_[6] * m11 + M_[10] * m21 + M_[14] * m31;
  M[10] = M_[2] * m02 + M_[6] * m12 + M_[10] * m22 + M_[14] * m32;
  M[11] = M_[2] * m03 + M_[6] * m13 + M_[10] * m23 + M_[14] * m33;
  // Fourth row.
  M[12] = M_[3] * m00 + M_[7] * m10 + M_[11] * m20 + M_[15] * m30;
  M[13] = M_[3] * m01 + M_[7] * m11 + M_[11] * m21 + M_[15] * m31;
  M[14] = M_[3] * m02 + M_[7] * m12 + M_[11] * m22 + M_[15] * m32;
  M[15] = M_[3] * m03 + M_[7] * m13 + M_[11] * m23 + M_[15] * m33;
}
