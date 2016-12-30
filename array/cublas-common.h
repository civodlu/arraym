#pragma once

DECLARE_NAMESPACE_NLL

#define CHECK_CUDA(expr) \
   ensure((expr) == CUBLAS_STATUS_SUCCESS, "CUBLAS failed!");

#define INIT_AND_CHECK_CUDA(expr) \
   {config.init(); ensure((expr) == CUBLAS_STATUS_SUCCESS, "CUBLAS failed!");}

// do a CUDA function call. Always make sure the init was run beforehand
#define CUDA_PASSED(expr)                                 \
     config.init(), (!((expr) == CUBLAS_STATUS_SUCCESS)); \

namespace blas
{
   namespace detail
   {
      inline cublasOperation_t is_transposed(CBLAS_ORDER matrixOrder, CBLAS_TRANSPOSE transp)
      {
         // NVBLAS is column major
         if (matrixOrder == CBLAS_ORDER::CblasColMajor)
         {
            if (transp == CBLAS_TRANSPOSE::CblasNoTrans)
            {
               return cublasOperation_t::CUBLAS_OP_N;
            }
            return cublasOperation_t::CUBLAS_OP_T;
         }
         else{
            if (transp == CBLAS_TRANSPOSE::CblasNoTrans)
            {
               return cublasOperation_t::CUBLAS_OP_T;
            }
            return cublasOperation_t::CUBLAS_OP_N;
         }
      }
   }
}

DECLARE_NAMESPACE_NLL_END
