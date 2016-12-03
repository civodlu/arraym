#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Simplify the std::enable_if expression so that it is readable
*/
template <class T, size_t N, class Config>
using Matrix_NaiveEnabled =
    typename std::enable_if<array_use_naive<Array<T, N, Config>>::value && is_matrix<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

template <class T, size_t N, class Config>
using Matrix_Enabled =
typename std::enable_if<is_matrix<Array<T, N, Config>>::value, Array<T, N, Config>>::type;


namespace details
{
/**
   @brief Matrix * Matrix operation

   Very slow version just for compatibility with unsual types not handled by BLAS... should probably always avoid it!
   */
template <class T, class Config, class Config2>
Matrix_NaiveEnabled<T, 2, Config> array_mul_array(const Array<T, 2, Config>& op1, const Array<T, 2, Config2>& op2)
{
   const size_t op2_sizex = op2.shape()[1];
   const size_t op1_sizex = op1.shape()[1];
   const size_t op1_sizey = op1.shape()[0];

   Array<T, 2, Config> m({op1_sizey, op2_sizex});
   for (size_t nx = 0; nx < op2_sizex; ++nx)
   {
      for (size_t ny = 0; ny < op1_sizey; ++ny)
      {
         T val = 0;
         for (size_t n = 0; n < op1_sizex; ++n)
         {
            val += op1(ny, n) * op2(n, nx);
         }
         m(ny, nx) = val;
      }
   }
   return m;
}
}

/**
 @brief Transpose a matrix
 */
template <class T, class Config>
Matrix_Enabled<T, 2, Config> transpose(const Array<T, 2, Config>& m)
{
   Array<T, 2, Config> r(m.sizex(), m.sizey());
   for (size_t nx = 0; nx < r.sizex(); ++nx)
   {
      for (size_t ny = 0; ny < r.sizey(); ++ny)
      {
         r(ny, nx) = m(nx, ny);
      }
   }
   return r;
}

DECLARE_NAMESPACE_END