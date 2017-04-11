#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using array_type = Array_row_major<float, 3>;
static const array_type::index_type shape{512, 513, 700};

static const std::string file = "d:/tmp/big.dat";

struct TestArrayPerformance
{
   static array_type create(const array_type::index_type& shape)
   {
      array_type array(shape);

      auto functor = [](const array_type::index_type&) { return array_type::value_type(rand() % 1000); };
      fill_index(array, functor);
      return array;
   }

   static array_type create_max(const array_type::index_type& shape)
   {
      array_type array(shape);

      auto functor = [](const array_type::value_type) { return array_type::value_type(rand() % 1000); };
      fill_value(array, functor);
      return array;
   }

   void test_iterator_single()
   {
      Timer timer;
      auto a                       = create(shape);
      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;
      ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(a, 1);

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

      auto a = create_max(shape);
      ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(a, 0);

      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;

      bool hasMoreElements = true;
      while (hasMoreElements)
      {
         array_type::value_type* ptr = 0;
         hasMoreElements             = iterator.accessMaxElements(ptr);
         for (NAMESPACE_NLL::ui32 n = 0; n < iterator.getNbElementsPerAccess(); ++n)
         {
            ptr[n] = ptr[n] * ptr[n];
         }
      }

      std::cout << "Max=" << timer.getElapsedTime() - constructionTime << std::endl;
   }

   void test_values_iteration()
   {
      srand(0);
      Timer timer;

      auto a = create_max(shape);
      
      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;

      for (auto& value : values(a))
      {
         value *= value;
      }

      std::cout << "Values=" << timer.getElapsedTime() - constructionTime << std::endl;

      std::cout << *(array_base_memory(a)+1) << std::endl;
   }

   
   void test_values_iteration_iter()
   {
      srand(0);
      Timer timer;

      auto a = create_max(shape);

      const float constructionTime = timer.getElapsedTime();
      std::cout << "ConstructionTime=" << constructionTime << std::endl;

      auto proxy = values(a);
      auto end = proxy.end();
      for (auto it = proxy.begin(); it != end; ++it)
      {
         *it *= *it;
      }

      std::cout << "Values=" << timer.getElapsedTime() - constructionTime << std::endl;

      std::cout << *(array_base_memory(a) + 1) << std::endl;
   }

   void test_values_vector()
   {
      size_t nb_iter = 10;
      std::vector<float> values_vec(100000000);
      for (auto& value : values_vec)
      {
         value = generateUniformDistribution(-1.0f, 1.0f);
      }

      float sum = 0;

      
      Timer timerStditeration;
      {
         for (size_t n = 0; n < nb_iter; ++n)
         {
            for (auto& value : values_vec)
            {
               sum += value;
            }
         }
      }
      std::cout << "timerStditeration=" << timerStditeration.getElapsedTime() << std::endl;

      Timer timerValuesIteration;
      {
         for (size_t n = 0; n < nb_iter; ++n)
         {
            for (auto& value : enumerate(values_vec))
            {
               sum += *value;
            }
         }
      }
      std::cout << "timerValuesIteration=" << timerValuesIteration.getElapsedTime() << std::endl;
      
      using array_type = Array<float, 1>;
      array_type af(array_type::Memory(vector1ui(values_vec.size()), &values_vec[0]));
      
      //using array_type = Array<float, 2>;
      //array_type af(array_type::Memory(vector2ui(values_vec.size(), 1), &values_vec[0]));

      Timer timerArrayIteration;
      {
         for (size_t n = 0; n < nb_iter; ++n)
         {
            
            for (auto& value : values(af))
            {
               sum += value;
            }
         }
      }
      std::cout << "timerArrayIteration=" << timerArrayIteration.getElapsedTime() << std::endl;

      std::cout << sum << std::endl;
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
      for (size_t n = 0; n < 100000000; ++n)
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

   void test_write_speed()
   {
      using array_type = Array<float, 3>;

      array_type array(512, 512, 600);
      float start = 0;
      fill_value(array, [&](const float&) { return ++start; });

      Timer timerWrite;
      {
         std::ofstream f(file.c_str(), std::ios::binary);
         array.write(f);
      }
      std::cout << "writeTimer=" << timerWrite.getElapsedTime() << std::endl;

      array_type array2;
      Timer timerRead;
      {
         std::ifstream f(file.c_str(), std::ios::binary);
         array2.read(f);
      }
      std::cout << "timerRead=" << timerRead.getElapsedTime() << std::endl;

      TESTER_ASSERT(array == array2);
   }

   void test_read_speed()
   {
      using array_type = Array<float, 3>;
      array_type array2;
      Timer timerRead;
      {
         std::ifstream f(file.c_str(), std::ios::binary);
         array2.read(f);
      }
      std::cout << "timerRead=" << timerRead.getElapsedTime() << std::endl;

      float start = 0;
      fill_value(array2, [&](const float& value) { TESTER_ASSERT(++start == value); return start; });
   }
};

TESTER_TEST_SUITE(TestArrayPerformance);
TESTER_TEST(test_values_vector);
/*
TESTER_TEST(test_range_dummy);
TESTER_TEST(test_range);
TESTER_TEST(test_iterator_single);
TESTER_TEST(test_iterator_max);
TESTER_TEST(test_values_iteration);
TESTER_TEST(test_values_iteration_iter);
TESTER_TEST(test_iterator_dummy);
TESTER_TEST(test_write_speed);
TESTER_TEST(test_read_speed);
*/
TESTER_TEST_SUITE_END();
