#pragma once

DECLARE_NAMESPACE_NLL

template <class T, size_t N, class Config>
using Matrix_BlasEnabled =
    typename std::enable_if<array_use_blas<Array<T, N, Config>>::value && is_matrix<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

namespace details
{
/**
    @brief Determine at runtime what is the actual memory order. It don't assume the  Matrix_column_major is actually column major
    as we may have modified the index mapper for some specific optimization (e.g., transpose or compatibility with numpy)
    */
template <class Array>
struct MatrixMemoryOrder_
{
   static CBLAS_ORDER compute(const Array&)
   {
      return CBLAS_ORDER::UnkwownMajor;
   }
};

template <class T, class Allocator>
struct MatrixMemoryOrder_<Matrix_column_major<T, Allocator>>
{
   using Array = Matrix_column_major<T, Allocator>;

   static CBLAS_ORDER compute(const Array& array)
   {
      const auto& stride = array.getMemory().getIndexMapper()._getPhysicalStrides();
      if (stride[0] < stride[1])
      {
         return CBLAS_ORDER::CblasColMajor;
      }
      return CBLAS_ORDER::CblasRowMajor;
   }
};

template <class T, class Allocator>
struct MatrixMemoryOrder_<Matrix_row_major<T, Allocator>>
{
   using Array = Matrix_row_major<T, Allocator>;

   static CBLAS_ORDER compute(const Array& array)
   {
      const auto& stride = array.getMemory().getIndexMapper()._getPhysicalStrides();
      if (stride[0] < stride[1])
      {
         return CBLAS_ORDER::CblasColMajor;
      }
      return CBLAS_ORDER::CblasRowMajor;
   }
};
}

template <class Array>
CBLAS_ORDER getMatrixMemoryOrder(const Array& array)
{
   return details::MatrixMemoryOrder_<Array>::compute(array);
}

/**
@brief Extract the leading dimension of a matrix

This only make sense for contiguous memory array. This specifies the offset of the next row or column (depending
on the memory order)
*/
template <class T, class Config>
blas::BlasInt leading_dimension( const Matrix_Enabled<T, 2, Config>& a )
{
   blas::BlasInt lda = 0;
   const auto memory_order_a = getMatrixMemoryOrder( a );
   const auto& stride_a = a.getMemory().getIndexMapper()._getPhysicalStrides();
   if ( memory_order_a == CBLAS_ORDER::CblasColMajor )
   {
      lda = stride_a[ 1 ];
      ensure( stride_a[ 0 ] == 1, "can't have stride != 1  for BLAS" );
   } else
   {
      lda = stride_a[ 0 ];
      ensure( stride_a[ 1 ] == 1, "can't have stride != 1  for BLAS " );
   }
   return lda;
}

namespace details
{
/**
@brief Compute opc = alpha * opa * opb + beta * opc

Only call this methods for BLAS supported types (float/double) with Matrix based arrays
*/
template <class T, class Config, class Config2, class Config3>
void gemm(bool trans_a, bool trans_b, T alpha, const Array<T, 2, Config>& opa, const Array<T, 2, Config2>& opb, T beta, Array<T, 2, Config3>& opc)
{
   const auto memory_order_a = getMatrixMemoryOrder(opa);
   const auto memory_order_b = getMatrixMemoryOrder(opb);
   const auto memory_order_c = getMatrixMemoryOrder(opc);

   ensure(memory_order_a == memory_order_b, "matrix must have the same memory order");
   ensure(memory_order_a == memory_order_c, "matrix must have the same memory order");
   ensure(memory_order_a != CBLAS_ORDER::UnkwownMajor, "unkown memory order!");
   const auto order = memory_order_a == CBLAS_ORDER::CblasRowMajor ? CBLAS_ORDER::CblasRowMajor : CBLAS_ORDER::CblasColMajor;
   
   const blas::BlasInt lda = leading_dimension<T, Config>( opa );
   const blas::BlasInt ldb = leading_dimension<T, Config2>( opb );
   const blas::BlasInt ldc = leading_dimension<T, Config3>( opc );

   const auto trans_a_blas = trans_a ? blas::CblasTrans : blas::CblasNoTrans;
   const auto trans_b_blas = trans_b ? blas::CblasTrans : blas::CblasNoTrans;

   const auto m = rows(opa, trans_a);
   const auto n = columns(opb, trans_b);
   const auto k = columns(opa, trans_a);

   ensure(opc.rows() == m, "must be a opa.rows() * opb.columns()");

   core::blas::gemm<T>(order, trans_a_blas, trans_b_blas,
                       m, n, k,
                       alpha, &opa(0, 0), lda, &opb(0, 0), ldb, beta, &opc(0, 0), ldc);
}

template <class T, class Config, class Config2, class Config3>
void gemm( T alpha, const Array<T, 2, Config>& opa, const Array<T, 2, Config2>& opb, T beta, Array<T, 2, Config3>& opc )
{
   gemm( false, false, alpha, opa, opb, beta, opc );
}

template <class T, class Config, class Config2>
Matrix_BlasEnabled<T, 2, Config> array_mul_array(const Array<T, 2, Config>& opa, const Array<T, 2, Config2>& opb)
{
   Array<T, 2, Config> opc(opa.rows(), opb.columns());
   gemm(static_cast<T>(1), opa, opb, static_cast<T>(0), opc);
   return opc;
}
}

template <class T, class Config>
blas::BlasInt rows( const Array<T, 2, Config>& a, bool a_transposed = false )
{
   if ( !a_transposed )
   {
      return static_cast<blas::BlasInt>(a.rows());
   } else
   {
      return static_cast<blas::BlasInt>( a.columns() );
   }
}

template <class T, class Config>
blas::BlasInt columns( const Array<T, 2, Config>& a, bool a_transposed = false )
{
   if ( !a_transposed )
   {
      return static_cast<blas::BlasInt>( a.columns() );
   } else
   {
      return static_cast<blas::BlasInt>( a.rows() );
   }
}

DECLARE_NAMESPACE_NLL_END