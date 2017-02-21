#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

/**
 @tparam Op must be callable with (pointer_type1 a1, ui32 stride_a1, const_pointer_type2 a2, ui32 stride_a2, const_pointer_type3 a3, ui32 stride_a3, ui32 nb_elements)
 */
DECLARE_NAMESPACE_NLL
template <class T1, class T2, class T3, size_t N, class Config1, class Config2, class Config3, class Op, typename = typename std::enable_if<IsArrayLayoutLinear<Array<T1, N, Config1>>::value>::type>
void iterate_array_constarray_constarray(Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2, const Array<T3, N, Config3>& a3, Op& op)
{
   ensure(a1.shape() == a2.shape(), "must have the same shape!");
   ensure(a1.shape() == a3.shape(), "must have the same shape!");
   ensure(same_data_ordering(a1, a2), "must have the same ordering!");
   ensure(same_data_ordering(a1, a3), "must have the same ordering!");

   using array_type1 = Array<T1, N, Config1>;
   using array_type2 = Array<T2, N, Config2>;
   using array_type3 = Array<T3, N, Config3>;
   
   using pointer_type1 = typename array_type1::pointer_type;
   using const_pointer_type2 = typename array_type2::const_pointer_type;
   using const_pointer_type3 = typename array_type3::const_pointer_type;
   ArrayProcessor_contiguous_byMemoryLocality<array_type1>      processor_a1(a1, 0);
   ConstArrayProcessor_contiguous_byMemoryLocality<array_type2> processor_a2(a2, 0);
   ConstArrayProcessor_contiguous_byMemoryLocality<array_type3> processor_a3(a3, 0);

   static_assert(is_callable_with<Op, pointer_type1, ui32, const_pointer_type2, ui32, const_pointer_type3, ui32, ui32>::value, "Op is not callable with the correct arguments!");

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      pointer_type1 ptr_a1(nullptr);
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);

      const_pointer_type2 ptr_a2(nullptr);
      processor_a2.accessMaxElements(ptr_a2);

      const_pointer_type2 ptr_a3(nullptr);
      processor_a3.accessMaxElements(ptr_a3);

      op(ptr_a1, processor_a1.stride(), ptr_a2, processor_a2.stride(), ptr_a3, processor_a3.stride(), processor_a1.getNbElementsPerAccess());
   }
}

namespace details
{
   template <class T1, class T2, class T3, class F>
   void apply_fun2_array_strided(T1* output, ui32 output_stride, const T2* input1, ui32 input1_stride, const T3* input2, ui32 input2_stride, ui32 nb_elements, F f)
   {
      static_assert(is_callable_with<F, T2, T3>::value, "Op is not callable with the correct arguments!");

      const T1* output_end = output + output_stride * nb_elements;
      for (; output != output_end; output += output_stride, input1 += input1_stride, input2 += input2_stride)
      {
         *output = f(*input1, *input2);
      }
   };

   template <class Output>
   struct ApplyBinOp
   {
      template <class T1, class T2, class Config1, class Config2, size_t N, class Op>
      typename Array<T1, N, Config1>::template rebind<Output>::other operator()(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2, Op& op)
      {
         NLL_FAST_ASSERT(a1.shape() == a2.shape(), "must have the same shape!");
         using result_type = typename Array<T1, N, Config1>::template rebind<ui8>::other;

         auto f = [&](ui8* output, ui32 output_stride, const T1* input1, ui32 input1_stride, const T2* input2, ui32 input2_stride, ui32 nb_elements)
         {
            details::apply_fun2_array_strided(output, output_stride, input1, input1_stride, input2, input2_stride, nb_elements, op);
         };

         result_type result(a1.shape());
         iterate_array_constarray_constarray(result, a1, a2, f);
         return result;
      }
   };

   template <class Output>
   struct ApplyUnaryOp
   {
      template <class T1, class Config1, size_t N, class Op>
      typename Array<T1, N, Config1>::template rebind<Output>::other operator()(const Array<T1, N, Config1>& a1, Op& op)
      {
         using result_type = typename Array<T1, N, Config1>::template rebind<ui8>::other;

         auto f = [&](Output* output, ui32 output_stride, const T1* input1, ui32 input1_stride, ui32 nb_elements)
         {
            details::apply_fun_array_strided(output, output_stride, input1, input1_stride, nb_elements, op);
         };

         result_type result(a1.shape());
         iterate_array_constarray(result, a1, f);
         return result;
      }
   };
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::less<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<(const Array<T1, N, Config1>& a1, T2 value)
{
   auto value_converted = static_cast<T1>(value);

   auto op = [&](T1 v)->ui8
   {
      return v < value_converted;
   };
   return details::ApplyUnaryOp<ui8>()(a1, op);
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<(T2 value, const Array<T1, N, Config1>& a1)
{
   auto value_converted = static_cast<T1>(value);

   auto op = [&](T1 v)->ui8
   {
      return v > value_converted;
   };
   return details::ApplyUnaryOp<ui8>()(a1, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<=(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::less_equal<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::greater<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>=(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::greater_equal<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator&(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::logical_and<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator|(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::logical_or<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other equal_elementwise(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::equal_to<>();
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other different_elementwise(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   auto op = std::not_equal_to<>;
   return details::ApplyBinOp<ui8>()(a1, a2, op);
}

template <class T1, class Config1, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator!(const Array<T1, N, Config1>& a1)
{
   std::logical_not<> op;
   return details::ApplyUnaryOp<ui8>()(a1, op);
}


DECLARE_NAMESPACE_NLL_END

struct TestArrayLogicalOp
{
   void test_lessthan()
   {
      test_lessthan_impl<Array<float, 2>>();
      test_lessthan_impl<Array_column_major<float, 2>>();
      test_lessthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
                 5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
                 50, -10, 9 };

      auto r = array1 < array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_lessthan_value()
   {
      test_lessthan_value_impl<Array<float, 2>>();
      test_lessthan_value_impl<Array_column_major<float, 2>>();
      test_lessthan_value_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessthan_value_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 3 };

      auto r = array1 < 4;
      auto r2 = 4 < array1;

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 1);
      TESTER_ASSERT(norm2(r2 - !r) == 0);
   }

   void test_greaterthan()
   {
      test_greaterthan_impl<Array<float, 2>>();
      test_greaterthan_impl<Array_column_major<float, 2>>();
      test_greaterthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 > array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_greaterequalthan()
   {
      test_greaterequalthan_impl<Array<float, 2>>();
      test_greaterequalthan_impl<Array_column_major<float, 2>>();
      test_greaterequalthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterequalthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 >= array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_lessequalthan()
   {
      test_lessequalthan_impl<Array<float, 2>>();
      test_lessequalthan_impl<Array_column_major<float, 2>>();
      test_lessequalthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessequalthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 <= array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_and()
   {
      test_and_impl<Array<char, 2>>();
      test_and_impl<Array_column_major<int, 2>>();
      test_and_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_and_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
                 1, 1, 0 };

      array_type array2(3, 2);
      array2 = { 1, 0, 1,
                 0, 1, 0 };

      auto r = array1 & array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_or()
   {
      test_or_impl<Array<char, 2>>();
      test_or_impl<Array_column_major<int, 2>>();
      test_or_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_or_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
         1, 1, 0 };

      array_type array2(3, 2);
      array2 = { 1, 0, 1,
         0, 1, 0 };

      auto r = array1 | array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_not()
   {
      test_not_impl<Array<char, 2>>();
      test_not_impl<Array_column_major<int, 2>>();
      test_not_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_not_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
                 1, 1, 0 };

      auto r = !array1;
      
      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_equal()
   {
      test_equal_impl<Array<float, 2>>();
      test_equal_impl<Array_column_major<float, 2>>();
      test_equal_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_equal_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 4 };

      auto r = equal_elementwise(array1, array2);
      
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 0);
   }

   template <class array_type>
   void test_notequal_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 4 };

      auto r = different_elementwise(array1, array2);

      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
   }
};

TESTER_TEST_SUITE(TestArrayLogicalOp);
TESTER_TEST(test_lessthan_value);
TESTER_TEST(test_lessthan);
TESTER_TEST(test_lessequalthan);
TESTER_TEST(test_greaterthan);
TESTER_TEST(test_greaterequalthan);
TESTER_TEST(test_lessequalthan);
TESTER_TEST(test_equal);
TESTER_TEST(test_or);
TESTER_TEST(test_and);
TESTER_TEST(test_not);
TESTER_TEST_SUITE_END();