#include "wrapper-cublas.h"


#ifdef WITH_CUDA
#pragma warning(disable:4505) // unreferenced local function

#include <cublasXt.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace detail
{
   cublasOperation_t is_transposedXt(CBLAS_ORDER matrixOrder, CBLAS_TRANSPOSE transp)
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

#define CHECK_CUDA(expr) \
   ensure((expr) == CUBLAS_STATUS_SUCCESS, "CUBLAS failed!");

#define CUDA_PASSED(expr) \
   (!((expr) == CUBLAS_STATUS_SUCCESS));


   /**
    @brief Takes care of the initialization.
    
    @Note that the first call to init() is slow (about 0.5sec) and could bias benchmarks!
    */
   class CublasXtConfig
   {
   public:
      CublasXtConfig()
      {
      }

      void init(int device = -1)
      {
         if (_handle == nullptr)
         {
            CHECK_CUDA(cublasXtCreate(&_handle));
            if (device == -1)
            {
               device = findCudaDevice(0, (const char **)0);
            }
            int devices[1] = { device };
            CHECK_CUDA(cublasXtDeviceSelect(_handle, 1, devices));
         }
      }

      ~CublasXtConfig()
      {
         if (_handle != nullptr)
         {
            CHECK_CUDA(cublasXtDestroy(_handle));
         }
      }

      cublasXtHandle_t handle() const
      {
         return _handle;
      }

   private:
      cublasXtHandle_t _handle = nullptr;
   };

   CublasXtConfig config;

   BlasInt sgemm_cublasxt(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
      const BlasReal beta, BlasReal* C, const BlasInt ldc)
   {
      if (matrixOrder == CBLAS_ORDER::CblasRowMajor)
      {
         // CUDA is column major.
         // TODO for now just don't use CUDA, if we really want speed, we want to avoid the transpose anyway...
         //
         // cublas works on column-major style only, so we need to transpose the result if C is row major
         // http://stackoverflow.com/questions/15458552/what-is-the-most-efficient-way-to-transpose-a-matrix-in-cuda
         // cublasSdgmm
         return -1;
      }

      if (M * N * K < 10000)
      {
         // TODO check speed!
         return -1;
      }

      config.init();
      return CUDA_PASSED(cublasXtSgemm(config.handle(), is_transposedXt(matrixOrder, TransA), is_transposedXt(matrixOrder, TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc));
   }

   BlasInt dgemm_cublasxt(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
      const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc)
   {
      if (matrixOrder == CBLAS_ORDER::CblasRowMajor)
      {
         // CUDA is column major.
         // TODO for now just don't use CUDA, if we really want speed, we want to avoid the transpose anyway...
         //
         // cublas works on column-major style only, so we need to transpose the result if C is row major
         // http://stackoverflow.com/questions/15458552/what-is-the-most-efficient-way-to-transpose-a-matrix-in-cuda
         // cublasSdgmm
         return -1;
      }

      if (M * N * K < 10000)
      {
         // TODO check speed!
         return -1;
      }

      config.init();
      return CUDA_PASSED(cublasXtDgemm(config.handle(), is_transposedXt(matrixOrder, TransA), is_transposedXt(matrixOrder, TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc));
   }
}
}

DECLARE_NAMESPACE_NLL_END

#endif