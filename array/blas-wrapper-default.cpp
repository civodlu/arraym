#include <array/forward.h>
#include "wrapper-cblas.h"
#include "wrapper-cublas.h"
#include "wrapper-cublas-xt.h"

DECLARE_NAMESPACE_NLL

namespace blas
{
#ifdef WITH_CUDA
const size_t cuda_01 = BlasDispatcher::instance().registerFunction<details::BlasFunction::saxpy>(detail::saxpy_cublas, "saxpy_cublas");
const size_t cuda_02 = BlasDispatcher::instance().registerFunction<details::BlasFunction::daxpy>(detail::daxpy_cublas, "daxpy_cublas");

const size_t cuda_03 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgemm>(detail::sgemm_cublas, "sgemm_cublas");
const size_t cuda_04 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgemm>(detail::dgemm_cublas, "dgemm_cublas");

const size_t cuda_05 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sscal>(detail::sscal_cublas, "sscal_cublas");
const size_t cuda_06 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dscal>(detail::dscal_cublas, "dscal_cublas");

const size_t cuda_07 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sdot>(detail::sdot_cublas, "sdot_cublas");
const size_t cuda_08 = BlasDispatcher::instance().registerFunction<details::BlasFunction::ddot>(detail::ddot_cublas, "ddot_cublas");

const size_t cuda_13 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sasum>(detail::sasum_cublas, "sasum_cublas");
const size_t cuda_14 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dasum>(detail::dasum_cublas, "dasum_cublas");

const size_t cuda_15 = BlasDispatcher::instance().registerFunction<details::BlasFunction::snrm2>(detail::snrm2_cublas, "snrm2_cublas");
const size_t cuda_16 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dnrm2>(detail::dnrm2_cublas, "dnrm2_cublas");

const size_t cuda_09 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sger>(detail::sger_cublas, "sger_cublas");
const size_t cuda_10 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dger>(detail::dger_cublas, "dger_cublas");

const size_t cudaxt_01 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgemm>(detail::sgemm_cublasxt, "sgemm_cublasxt");
const size_t cudaxt_02 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgemm>(detail::dgemm_cublasxt, "dgemm_cublasxt");
#endif

#ifdef WITH_OPENBLAS
const size_t f2c_01 = BlasDispatcher::instance().registerFunction<details::BlasFunction::saxpy>(detail::saxpy_cblas, "saxpy_cblas");
const size_t f2c_02 = BlasDispatcher::instance().registerFunction<details::BlasFunction::daxpy>(detail::daxpy_cblas, "daxpy_cblas");

const size_t f2c_03 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgemm>(detail::sgemm_cblas, "sgemm_cblas");
const size_t f2c_04 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgemm>(detail::dgemm_cblas, "dgemm_cblas");

const size_t f2c_05 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sscal>(detail::sscal_cblas, "sscal_cblas");
const size_t f2c_06 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dscal>(detail::dscal_cblas, "dscal_cblas");

const size_t f2c_07 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sdot>(detail::sdot_cblas, "sdot_cblas");
const size_t f2c_08 = BlasDispatcher::instance().registerFunction<details::BlasFunction::ddot>(detail::ddot_cblas, "ddot_cblas");

const size_t f2c_13 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sasum>(detail::sasum_cblas, "sasum_cblas");
const size_t f2c_14 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dasum>(detail::dasum_cblas, "dasum_cblas");

const size_t f2c_15 = BlasDispatcher::instance().registerFunction<details::BlasFunction::snrm2>(detail::snrm2_cblas, "snrm2_cblas");
const size_t f2c_16 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dnrm2>(detail::dnrm2_cblas, "dnrm2_cblas");

const size_t f2c_09 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sger>(detail::sger_cblas, "sger_cblas");
const size_t f2c_10 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dger>(detail::dger_cblas, "dger_cblas");

const size_t f2c_11 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgemv>(detail::sgemv_cblas, "sgemv_cblas");
const size_t f2c_12 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgemv>(detail::dgemv_cblas, "dgemv_cblas");

//
// LAPACK
//
const size_t lapack_01 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgetrf>(detail::sgetrf_cblas, "sgetrf_cblas");
const size_t lapack_02 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgetrf>(detail::dgetrf_cblas, "dgetrf_cblas");

const size_t lapack_03 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgetri>(detail::sgetri_cblas, "sgetri_cblas");
const size_t lapack_04 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgetri>(detail::dgetri_cblas, "dgetri_cblas");

const size_t lapack_05 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgesdd>(detail::sgesdd_cblas, "sgesdd_cblas");
const size_t lapack_06 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgesdd>(detail::dgesdd_cblas, "dgesdd_cblas");

const size_t lapack_07 = BlasDispatcher::instance().registerFunction<details::BlasFunction::slaswp>(detail::slaswp_cblas, "slaswp_cblas");
const size_t lapack_08 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dlaswp>(detail::dlaswp_cblas, "dlaswp_cblas");

const size_t lapack_09 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgetrs>(detail::sgetrs_cblas, "sgetrs_cblas");
const size_t lapack_10 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgetrs>(detail::dgetrs_cblas, "dgetrs_cblas");

const size_t lapack_11 = BlasDispatcher::instance().registerFunction<details::BlasFunction::sgels>(detail::sgels_cblas, "sgels_cblas");
const size_t lapack_12 = BlasDispatcher::instance().registerFunction<details::BlasFunction::dgels>(detail::dgels_cblas, "dgels_cblas");

#endif
}
DECLARE_NAMESPACE_NLL_END
