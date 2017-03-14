#pragma once

#include <chrono>

class Timer
{
public:
   /**
   @brief Instanciate the timer and start it
   */
   Timer()
   {
      start();
      _end = _start;
   }

   /**
   @brief restart the timer
   */
   void start()
   {
      _start = std::chrono::high_resolution_clock::now();
   }

   /**
   @brief end the timer, return the time in seconds spent
   */
   void end()
   {
      _end = std::chrono::high_resolution_clock::now();
   }

   /**
   @brief get the current time since the begining, return the time in seconds spent.
   */
   float getElapsedTime() const
   {
      auto c = std::chrono::high_resolution_clock::now();
      return toSeconds(_start, c);
   }

   /**
   @brief return the time in seconds spent since between starting and ending the timer. The timer needs to be ended before calling it.
   */
   float getTime() const
   {
      return toSeconds(_start, _end);
   }

private:
   static float toSeconds(const std::chrono::high_resolution_clock::time_point& start, const std::chrono::high_resolution_clock::time_point& end)
   {
      const auto c = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      return static_cast<float>(c / 1000.0f);
   }

private:
   std::chrono::high_resolution_clock::time_point _start;
   std::chrono::high_resolution_clock::time_point _end;
};

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
