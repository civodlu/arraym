#pragma once

#include <array/array-api.h>
#include <array/config.h>
#include <array/traits.h>
#include "wrapper-common.h"

#pragma warning(disable : 4503) // decorated names too long

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace details
{
enum BlasFunction
{
   // BLAS Level 1
   saxpy = 0,
   daxpy,
   sasum,
   dasum,
   sdot,
   ddot,
   snrm2,
   dnrm2,

   // BLAS Level 2/3
   sgemm,
   dgemm,
   sscal,
   dscal,
   sger,
   dger,
   sgemv,
   dgemv,

   // LAPACK
   sgetrf,
   dgetrf,
   sgetri,
   dgetri,

   sgesdd,
   dgesdd,

   slaswp,
   dlaswp,

   sgetrs,
   dgetrs,

   sgels,
   dgels,
};

/**
      Record all the BLAS implementations (e.g., CPU, GPU) and run profiling to determine the best BLAS implementation for a given function call & arguments
      */
class ARRAY_API BlasDispatcherImpl
{
public:
   // these are the expected BLAS & LAPACK interface
   typedef std::function<void(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy)> saxpy_t;
   typedef std::function<void(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy)> daxpy_t;
   typedef std::function<BlasReal(const BlasInt n, const BlasReal* x, const BlasInt incx)> sasum_t;
   typedef std::function<BlasDoubleReal(const BlasInt n, const BlasDoubleReal* x, const BlasInt incx)> dasum_t;
   typedef std::function<BlasReal(const BlasInt N, const BlasReal* X, const BlasInt incX)> snrm2_t;
   typedef std::function<BlasDoubleReal(const BlasInt N, const BlasDoubleReal* X, const BlasInt incX)> dnrm2_t;

   typedef std::function<void(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                              const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
                              const BlasReal beta, BlasReal* C, const BlasInt ldc)>
       sgemm_t;
   typedef std::function<void(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                              const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B,
                              const BlasInt ldb, const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc)>
       dgemm_t;

   typedef std::function<void(const BlasInt N, const BlasReal alpha, BlasReal* X, const BlasInt incX)> sscal_t;
   typedef std::function<void(const BlasInt N, const BlasDoubleReal alpha, BlasDoubleReal* X, const BlasInt incX)> dscal_t;

   typedef std::function<BlasReal(const BlasInt N, const BlasReal* x, const BlasInt incX, const BlasReal* y, const BlasInt incY)> sdot_t;
   typedef std::function<BlasDoubleReal(const BlasInt N, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY)> ddot_t;

   typedef std::function<void(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* x, const BlasInt incX,
                              const BlasReal* y, const BlasInt incY, BlasReal* A, const BlasInt lda)>
       sger_t;
   typedef std::function<void(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha, const BlasDoubleReal* x,
                              const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY, BlasDoubleReal* A, const BlasInt lda)>
       dger_t;

   typedef std::function<void(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasReal alpha,
                              const BlasReal* A, const BlasInt lda, const BlasReal* x, const BlasInt incX, const BlasReal beta, BlasReal* y,
                              const BlasInt incY)>
       sgemv_t;
   typedef std::function<void(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha,
                              const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal beta,
                              BlasDoubleReal* y, const BlasInt incY)>
       dgemv_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasInt* ipiv)> sgetrf_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt* ipiv)> dgetrf_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, const BlasInt* ipiv)> sgetri_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv)> dgetri_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasReal* s, BlasReal* u, BlasInt ldu,
                                 BlasReal* vt, BlasInt ldvt)>
       sgesdd_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* s, BlasDoubleReal* u,
                                 BlasInt ldu, BlasDoubleReal* vt, BlasInt ldvt)>
       dgesdd_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)>
       slaswp_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)>
       dlaswp_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasReal* a, BlasInt lda, const BlasInt* ipiv,
                                 BlasReal* b, BlasInt ldb)>
       sgetrs_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv,
                                 BlasDoubleReal* b, BlasInt ldb)>
       dgetrs_t;

   typedef std::function<BlasInt(CBLAS_ORDER matrix_order, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasReal* a, BlasInt lda, BlasReal* b, BlasInt ldb)>
       sgels_t;
   typedef std::function<BlasInt(CBLAS_ORDER matrix_order, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* b,
                                 BlasInt ldb)>
       dgels_t;

private:
   typedef std::tuple<std::vector<saxpy_t>, std::vector<daxpy_t>, std::vector<sasum_t>, std::vector<dasum_t>, std::vector<sdot_t>, std::vector<ddot_t>,
                      std::vector<snrm2_t>, std::vector<dnrm2_t>, std::vector<sgemm_t>, std::vector<dgemm_t>, std::vector<sscal_t>, std::vector<dscal_t>,
                      std::vector<sger_t>, std::vector<dger_t>, std::vector<sgemv_t>, std::vector<dgemv_t>,

                      // LAPACK
                      std::vector<sgetrf_t>, std::vector<dgetrf_t>, std::vector<sgetri_t>, std::vector<dgetri_t>, std::vector<sgesdd_t>, std::vector<dgesdd_t>,
                      std::vector<slaswp_t>, std::vector<dlaswp_t>, std::vector<sgetrs_t>, std::vector<dgetrs_t>, std::vector<sgels_t>, std::vector<dgels_t>>
       Functions;

   typedef std::array<std::vector<std::string>, std::tuple_size<Functions>::value> FunctionIds;

   Functions _functions;
   FunctionIds _functionIds;

public:
   BlasDispatcherImpl(const std::string& path = "blasConfiguration.txt");

   template <int F>
   using function_t = typename std::tuple_element<F, Functions>::type::value_type;

   template <int F>
   using functions_t = typename std::tuple_element<F, Functions>::type;

   template <int F>
   using function_return_t = typename function_traits<function_t<F>>::return_type;

public:
   template <BlasFunction F>
   functions_t<F>& get()
   {
      auto& functions = std::get<F>(_functions);
      return functions;
   }

   template <BlasFunction F>
   const functions_t<F>& get() const
   {
      auto& functions = std::get<F>(_functions);
      return functions;
   }

   template <BlasFunction F, typename... Args>
   function_return_t<F> call(Args&&... args)
   {
      NLL_FAST_ASSERT(std::get<F>(_functions).size() > 0, "no registered BLAS wrapper!");
      static_assert(is_callable_with<function_t<F>, Args...>::value, "Expected arguments do not match the provided arguments");
      return std::get<F>(_functions)[0](std::forward<Args>(args)...); // @TODO find best dispatch!
   }

   template <BlasFunction F>
   size_t registerFunction(function_t<F> f, const std::string& functionId)
   {
      std::get<F>(_functions).push_back(f);
      _functionIds[F].push_back(functionId);
      return _functionIds[F].size() - 1;
   }

   /**
         @brief Calibrate the registered BLAS functions to be called given a specific configuration
         */
   void runBenchmark();
   void readConfiguration(const std::string& path = "blasConfiguration.txt");
   void writeConfiguration(const std::string& path = "blasConfiguration.txt") const;
};
}

class ARRAY_API BlasDispatcher
{
public:
   static details::BlasDispatcherImpl& instance();
};
}
DECLARE_NAMESPACE_NLL_END