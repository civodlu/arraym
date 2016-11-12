#pragma once

#include <type_traits>
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <stdint.h>

#include <array/config.h>

DECLARE_NAMESPACE_NLL
using ui32 = uint32_t;
DECLARE_NAMESPACE_END

#include <array/array-api.h>
#include "op.h"
#include "static-vector.h"
#include "static-vector-math.h"
#include "static-vector-op.h"
#include "traits.h"
#include "index-mapper.h"
#include "memory.h"
#include "array-traits.h"
#include "array.h"
#include "array-processor.h"