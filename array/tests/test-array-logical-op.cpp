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
}

using ui8 = unsigned char;

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<(const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2)
{
   NLL_FAST_ASSERT(a1.shape() == a2.shape(), "must have the same shape!");
   using result_type = typename Array<T1, N, Config1>::template rebind<ui8>::other;

   auto f = [](ui8* output, ui32 output_stride, const T1* input1, ui32 input1_stride, const T2* input2, ui32 input2_stride, ui32 nb_elements)
   {
      details::apply_fun2_array_strided(output, output_stride, input1, input1_stride, input2, input2_stride, nb_elements, std::less<>());
   };

   result_type result(a1.shape());
   iterate_array_constarray_constarray(result, a1, a2, f);
   return result;
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
      std::cout << r2 << std::endl;
   }
};

TESTER_TEST_SUITE(TestArrayLogicalOp);
TESTER_TEST(test_lessthan);
TESTER_TEST_SUITE_END();