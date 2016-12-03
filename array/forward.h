#pragma once

#include <array/config.h>

DECLARE_NAMESPACE_NLL
using ui32 = uint32_t;
DECLARE_NAMESPACE_END

#include <array/array-api.h>
#include "op-naive.h"
#include "static-vector.h"
#include "static-vector-math.h"
#include "static-vector-op.h"
#include "traits.h"
#include "index-mapper.h"
#include "memory-contiguous.h"
#include "memory-slice.h"
#include "array-traits.h"
#include "array.h"
#include "array-processor.h"
#include "array-op-naive.h"
#include "array-op-blas.h"

#include "wrapper-common.h"
#include "matrix-op-naive.h"
#include "matrix-op-blas.h"
#include "matrix-op-blas-inv.h"

#include "blas-wrapper.h"
#include "array-exp.h"
#include "array-noexp.h"