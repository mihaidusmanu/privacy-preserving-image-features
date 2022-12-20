#include <Eigen/Dense>

#include "basic_types.h"

using RowVectorType = Eigen::Matrix<ScalarType, 1, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
