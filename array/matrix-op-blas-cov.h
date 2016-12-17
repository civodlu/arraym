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
   const auto matrixOrder = getMatrixMemoryOrder(pointsByRow);

   matrix_type c({ columns, columns });
   const auto lda = leading_dimension<T, Config>(pointsByRow);
   const auto ldb = leading_dimension<T, Config>(pointsByRow);
   const auto ldc = leading_dimension<T, Config>(c);

   core::blas::gemm<T>(matrixOrder, blas::CBLAS_TRANSPOSE::CblasTrans, blas::CBLAS_TRANSPOSE::CblasNoTrans,
      columns, columns, rows,
      (T)(1.0 / (rows - 1)),
      &pointsByRow(0, 0),
      lda,
      &pointsByRow(0, 0),
      ldb,
      0,
      &c(0, 0),
      ldc);
   return c;
}

/**
@brief Compute the covariance a of a set of points (one by row)

pointsByRow is normalized to have zero mean. Compute X^tX using GEMM. Require a copy of <pointsByRow>
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> cov(const Array<T, 2, Config>& pointsByRow)
{
   const auto mean_row = mean(pointsByRow, 0);
   std::cout << mean_row << std::endl;
   return cov_zeroMean(pointsByRow);
   //pointsByRow
}

DECLARE_NAMESPACE_NLL_END