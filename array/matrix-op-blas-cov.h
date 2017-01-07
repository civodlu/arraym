#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Compute the covariance a of a set of points (one by row), where the mean of each column is 0

Compute X^tX using GEMM
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> cov_zeroMean(const Array<T, 2, Config>& pointsByRow)
{
   using matrix_type = Array<T, 2, Config>;

   const auto rows = static_cast<blas::BlasInt>(pointsByRow.rows());
   const auto columns = static_cast<blas::BlasInt>(pointsByRow.columns());

   matrix_type c({ columns, columns });
   details::gemm<T>(true, false, (T)(1.0 / (rows - 1)), pointsByRow, pointsByRow, 0, c);
   return c;
}

/**
@brief Compute the covariance a of a set of points (one by row)

pointsByRow is normalized to have zero mean. Compute X^tX using GEMM. Require a copy of <pointsByRow>
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> cov(const Array<T, 2, Config>& pointsByRow)
{
   using index_type = typename Array<T, 2, Config>::index_type;
   const auto mean_row = mean(pointsByRow, 0);

   // the mean has 1D shape [N], so we can't use repmat directly without transposing the matrix. Instead
   // express the mean row as a 2D array
   Array<T, 2, Config> mean_row_matrix;
   as_array(mean_row, index_type(1, mean_row.size()), mean_row_matrix);  // we don't know the memory type so stay as generic as possible

   const auto mean_mat = repmat(mean_row_matrix, index_type(pointsByRow.shape()[0], 1));
   auto pointsByRow_zeroMean = pointsByRow - mean_mat;
   return cov_zeroMean(pointsByRow_zeroMean);
}

DECLARE_NAMESPACE_NLL_END