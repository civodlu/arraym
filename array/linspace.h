#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Return evenly spaced numbers over a specified interval

linspace(1, 4, 5) = [1, 2, 3, 4, 5]
*/
template <class T>
Array<T, 1> linspace(T start, T stop, ui32 nb_steps)
{
   using array_type = Array<T, 1>;

   array_type result(nb_steps);

   // here we absolutely need to have a floating point value
   using t_float = typename PromoteFloating<T>::type;
   t_float scale = t_float(stop - start) / (nb_steps - 1);

   ui32 step = 0;
   fill_value(result, [&](T)
   {
      return static_cast<T>((step++) * scale) + start;
   });
   return result;
}

DECLARE_NAMESPACE_NLL_END