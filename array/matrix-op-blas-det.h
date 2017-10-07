#pragma once

DECLARE_NAMESPACE_NLL

template <class T, class Config>
double det(const Array<T, 2, Config>& a)
{
   Array<T, 2, Config> l;
   Array<T, 2, Config> u;
   ui32 nb_permutations;
   lu(a, l, u, &nb_permutations);

   double d = ((nb_permutations % 2) == 1) ? -1.0 : 1.0;
   for (ui32 n = 0; n < u.rows(); ++n)
   {
      d *= u(n, n);
   }

   return d;
}

DECLARE_NAMESPACE_NLL_END