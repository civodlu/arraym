#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using array_type = Array_row_major<float, 3>;
static const array_type::index_type shape{512, 513, 700};

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

struct TestArrayPerformance
{
   static array_type create(const array_type::index_type& shape)
   {
      array_type array(shape);

      auto functor = [](const array_type::index_type&) { return array_type::value_type(rand() % 1000); };
      fill(array, functor);
      return array;
   }

   void test_iterator_single()
   {
      Timer timer;
      auto a                       = create(shape);
      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;
      ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(a);

      bool hasMoreElements = true;
      while (hasMoreElements)
      {
         array_type::value_type* ptr = 0;
         hasMoreElements             = iterator.accessSingleElement(ptr);
         *ptr                        = *ptr * *ptr;
      }

      std::cout << "Single=" << timer.getElapsedTime() - constructionTime << std::endl;
   }

   void test_iterator_max()
   {
      Timer timer;

      auto a = create(shape);
      ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(a);

      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;

      bool hasMoreElements = true;
      while (hasMoreElements)
      {
         array_type::value_type* ptr = 0;
         hasMoreElements             = iterator.accessMaxElements(ptr);
         for (NAMESPACE_NLL::ui32 n = 0; n < iterator.getMaxAccessElements(); ++n)
         {
            ptr[n] = ptr[n] * ptr[n];
         }
      }

      std::cout << "Max=" << timer.getElapsedTime() - constructionTime << std::endl;
   }

   void test_iterator_dummy()
   {
      Timer timer;
      auto a = create(shape);

      array_type::value_type* ptr = &a(0, 0, 0);

      const auto y_stride = a.shape()[0];
      const auto z_stride = a.shape()[0] * a.shape()[1];

      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;
      for (ui32 z = 0; z < a.shape()[2]; ++z)
      {
         for (ui32 y = 0; y < a.shape()[1]; ++y)
         {
            for (ui32 x = 0; x < a.shape()[0]; ++x)
            {
               auto ptr_v = ptr + x + y * y_stride + z * z_stride;
               *ptr_v     = *ptr_v * *ptr_v;
            }
         }
      }

      std::cout << "Dummy=" << timer.getElapsedTime() - constructionTime << std::endl;
   }

   void test_range_dummy()
   {
      size_t f = 0;
      for ( size_t n = 0; n < 100000000; ++n )
      {
         f += n;
      }
      std::cout << f << std::endl;
   }

   void test_range()
   {
      size_t f = 0;
      for (auto n : range((size_t)100000000))
      {
         f += n;
      }
      std::cout << f << std::endl;
   }
};

TESTER_TEST_SUITE(TestArrayPerformance);
TESTER_TEST( test_range_dummy );
TESTER_TEST( test_range );
TESTER_TEST(test_iterator_single);
TESTER_TEST(test_iterator_max);
TESTER_TEST(test_iterator_dummy);
TESTER_TEST_SUITE_END();