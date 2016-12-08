#pragma once

namespace details
{
//
// TODO DO NOT USE RAND() and use thread local generators
//
struct UniformDistribution
{
   template <class T>
   static T generate(T min, T max, std::false_type UNUSED(isIntegral))
   {
      NLL_FAST_ASSERT(min <= max, "invalid range!");
      return static_cast<T>(rand()) / RAND_MAX * (max - min) + min;
   }

   template <class T>
   static T generate(T min, T max, std::true_type UNUSED(isIntegral))
   {
      NLL_FAST_ASSERT(min <= max, "invalid range!");
      const T interval = max - min + 1;
      if (interval == 0)
         return min;
      return (rand() % interval) + min;
   }
};
}

/**
@ingroup core
@brief generate a sample of a specific uniform distribution
@param min the min of the distribution, inclusive
@param max the max of the distribution, inclusive
@return a sample of this distribution
*/
template <class T>
T generateUniformDistribution(T min, T max)
{
   static_assert(std::is_arithmetic<T>::value, "must be a numeric type!");
   return ::details::UniformDistribution::generate<T>(min, max, std::is_integral<T>());
}