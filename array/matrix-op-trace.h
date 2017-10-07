#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief compute the trace of a matrix
*/
template <class T, class Config>
T trace(const Array<T, 2, Config>& m)
{
   ensure(m.shape()[0] == m.shape()[1], "must be square matrix!");

   T accum = T(0);
   for (size_t k = 0; k < m.shape()[0]; ++k)
   {
      accum += m({k, k});
   }
   return accum;
}

DECLARE_NAMESPACE_NLL_END
