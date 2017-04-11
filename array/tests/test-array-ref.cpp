#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayRef
{
   template <class array_type>
   static array_type create(const typename array_type::index_type& shape)
   {
      array_type array(shape);

      auto functor = [](const typename array_type::index_type&) { return typename array_type::value_type(rand() % 1000); };
      fill_index(array, functor);
      return array;
   }

   void test_ref_copy_array()
   {
      test_ref_copy_array_impl<Array<int, 2>>();
      test_ref_copy_array_impl<Array_column_major<float, 2>>();
      test_ref_copy_array_impl<Array_row_major_multislice<int, 2>>();
      test_ref_copy_array_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class Array>
   void test_ref_copy_array_impl()
   {
      Array a1 = create<Array>({13, 14});
      Array a2(2, 3);
      a2 = {10, 11, 12, 13, 14, 15};

      auto a1_cpy = a1;

      const auto origin = vector2ui(1, 2);
      const auto max    = origin + a2.shape() - 1;
      auto ref_a1       = a1(origin, max);
      ref_a1            = a2;

      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const bool is_inside = x >= origin[0] && x <= max[0] && y >= origin[1] && y <= max[1];
            if (is_inside)
            {
               TESTER_ASSERT(a2(vector2ui{x, y} - origin) == a1(x, y));
            }
            else
            {
               TESTER_ASSERT(a1(x, y) == a1_cpy(x, y));
            }
         }
      }

      a1_cpy(0, 0) = 42;
      TESTER_ASSERT(a1_cpy(0, 0) != a1(0, 0)); // no reference!
   }

   void test_ref_copy_value()
   {
      test_ref_copy_value_impl<Array<int, 2>>();
      test_ref_copy_value_impl<Array_column_major<float, 2>>();
      test_ref_copy_value_impl<Array_row_major_multislice<int, 2>>();
      test_ref_copy_value_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class Array>
   void test_ref_copy_value_impl()
   {
      Array a1                               = create<Array>({13, 14});
      const typename Array::value_type value = 42;

      auto a1_cpy = a1;

      const auto origin = vector2ui(5, 7);
      const auto max    = origin + vector2ui(3, 4);
      auto ref_a1       = a1(origin, max);
      ref_a1            = value;

      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const bool is_inside = x >= origin[0] && x <= max[0] && y >= origin[1] && y <= max[1];
            if (is_inside)
            {
               TESTER_ASSERT(value == a1(x, y));
            }
            else
            {
               TESTER_ASSERT(a1(x, y) == a1_cpy(x, y));
            }
         }
      }
   }

   void test_rebindMemory_contiguous()
   {
      {
         using Array_i  = Array<int, 2>;
         using Array_f  = Array<float, 2>;
         using Memory_f = Array_i::Memory::rebind<float>::other;

         Memory_f a1(Memory_f::index_type(4, 5));
         Array_f array(a1);
      }
   }

   void test_rebindMemory_slice()
   {
      {
         using Array_i  = Array_row_major_multislice<int, 2>;
         using Array_f  = Array_row_major_multislice<float, 2>;
         using Memory_f = Array_i::Memory::rebind<float>::other;

         Memory_f a1(Memory_f::index_type(4, 5));
         Array_f array(a1);
      }
   }

   void test_rebindArray_contiguous()
   {
      using Array_i  = Array_row_major<int, 2>;
      using Array_f  = Array_row_major<float, 2>;
      using Array_f2 = Array_i::rebind<float>::other;

      static_assert(std::is_same<Array_f, Array_f2>::value, "must be the same type!");
      Array_f2 a1(Array_f2::index_type(4, 5));
      Array_f array(a1);
   }

   void test_rebindArray_slice()
   {
      using Array_i  = Array_row_major_multislice<int, 2>;
      using Array_f  = Array_row_major_multislice<float, 2>;
      using Array_f2 = Array_i::rebind<float>::other;

      static_assert(std::is_same<Array_f, Array_f2>::value, "must be the same type!");
      Array_f2 a1(Array_f2::index_type(4, 5));
      Array_f array(a1);
   }

   void test_constSubArray()
   {
      test_constSubArray_impl<Array<int, 2>>();
      test_constSubArray_impl<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_constSubArray_impl()
   {
      using T = typename Array::value_type;
      Array a1(3, 2);
      a1 = {1, 2, 3, 4, 5, 6};

      auto m = a1.getMemory().asConst();

      auto a1_const = a1.asConst();
      TESTER_ASSERT(&a1_const(0, 0) == &a1(0, 0)); // there should be no copy here! We just see the data as const!
      TESTER_ASSERT(a1_const.shape() == a1.shape());

      T value  = 0;
      auto sum = [&](T const* a1_pointer, ui32 a1_stride, ui32 nb_elements) { value += details::norm2_naive_sqr(a1_pointer, a1_stride, nb_elements); };

      // use array on <cont T>, can't modify the array
      iterate_array(a1_const, sum);
      TESTER_ASSERT(value == 91);

      value = 0;
      iterate_constarray(a1_const, sum);
      TESTER_ASSERT(value == 91);
   }

   /*
   // TODO fix!
   void test_ref_copy_const_ref()
   {
      using array_type = Array<int, 1>;
      array_type a1(4);
      a1 = { 1, 2, 3, 4 };

      array_type a2(a1.shape());
      auto a2_ref = array_type::array_type_ref(a2);

      auto const_a1 = array_type::const_array_type_ref(a1.asConst());
      a2_ref = const_a1;

      TESTER_ASSERT(a2.shape() == a1.shape());
      TESTER_ASSERT(a2 == a1);
   }*/

   void test_copy_const_ref()
   {
      using array_type = Array<int, 1>;
      array_type a1(4);
      a1 = { 1, 2, 3, 4 };

      array_type a2;
      
      auto const_a1 = array_type::const_array_type_ref(a1.asConst());
      a2 = const_a1;

      TESTER_ASSERT(a2.shape() == a1.shape());
      TESTER_ASSERT(a2 == a1);
   }

   void test_copy_ref()
   {
      using array_type = Array<int, 1>;
      array_type a1(4);
      a1 = { 1, 2, 3, 4 };

      array_type a2;

      auto ref_a1 = array_type::array_type_ref(a1);
      a2 = ref_a1;

      TESTER_ASSERT(a2.shape() == a1.shape());
      TESTER_ASSERT(a2 == a1);
   }

   void test_copy_construct_ref()
   {
      // expected to copy there reference, not the whole array (important for boost.python asArray)
      using array_type = Array<int, 1>;
      array_type a1(4);
      a1 = { 1, 2, 3, 4 };

      array_type::array_type_ref ref(a1);
      array_type::array_type_ref ref2(ref);
      TESTER_ASSERT(ref.shape() == ref2.shape());
      TESTER_ASSERT(ref.shape() == a1.shape());

      // make sure we are pointing to the same array
      ref2[1] = 8;
      TESTER_ASSERT(ref[1] == 8);
      TESTER_ASSERT(a1[1] == 8);
   }
};

TESTER_TEST_SUITE(TestArrayRef);
TESTER_TEST(test_ref_copy_array);
TESTER_TEST(test_ref_copy_value);
TESTER_TEST(test_constSubArray);
TESTER_TEST(test_rebindMemory_contiguous);
TESTER_TEST(test_rebindMemory_slice);
TESTER_TEST(test_rebindArray_contiguous);
TESTER_TEST(test_rebindArray_slice);
//TESTER_TEST(test_ref_copy_const_ref);
TESTER_TEST(test_copy_const_ref);
TESTER_TEST(test_copy_ref);
TESTER_TEST(test_copy_construct_ref);
TESTER_TEST_SUITE_END();
