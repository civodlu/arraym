
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

DECLARE_NAMESPACE_NLL

namespace details
{
   template <class T, size_t N, class Config, ui32 dim>
   struct ArrayDimIterator
   {
   public:
      using array_type = Array<T, N, Config>;
      using array_ref_type = ArrayRef<T, N, Config>;
      using index_type = typename array_type::index_type;

      ArrayDimIterator(array_type* array, ui32 current) : _array(array), _current(current)
      {}

      ArrayDimIterator& operator++()
      {
         ++_current;
         return *this;
      }

      bool operator==(const ArrayDimIterator& other) const
      {
         NLL_FAST_ASSERT(other._array == _array, "Must be based on the same array!");
         return other._current == _current;
      }

      bool operator!=(const ArrayDimIterator& other) const
      {
         return !operator==(other);
      }

      array_ref_type operator*()
      {
         index_type min_index;
         index_type max_index = _array->shape() - 1;
         min_index[dim] = _current;
         max_index[dim] = _current;
         return (*_array)(min_index, max_index);
      }

   private:
      array_type* _array;
      ui32        _current;
   };

   template <class T, size_t N, class Config, ui32 dim>
   struct ArrayDimIterator_proxy
   {
      using array_iterator = ArrayDimIterator<T, N, Config, dim>;

      ArrayDimIterator_proxy(array_iterator begin, array_iterator end) : _begin(begin), _end(end)
      {}

      array_iterator begin() const
      {
         return _begin;
      }

      array_iterator end() const
      {
         return _end;
      }

   private:
      array_iterator _begin;
      array_iterator _end;
   };
}

template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<T, N, Config, 1> rows(Array<T, N, Config>& array)
{
   static const ui32 dim = 1;
   using proxy_type = details::ArrayDimIterator_proxy<T, N, Config, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<T, N, Config, 0> columns(Array<T, N, Config>& array)
{
   static const ui32 dim = 0;
   using proxy_type = details::ArrayDimIterator_proxy<T, N, Config, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}



DECLARE_NAMESPACE_NLL_END

struct TestArrayDimIterator
{
   void test_columns()
   {
      test_columns_impl<Array_row_major<float, 2>>();
      test_columns_impl<Array_column_major<float, 2>>();
      test_columns_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_columns_impl()
   {
      array_type a(3, 2);
      a = { 1, 2, 3,
            4, 5, 6 };

      int r = 0;
      for (auto row : rows(a))
      {
         TESTER_ASSERT(row.shape() == vector2ui(3, 1));
         TESTER_ASSERT(row(0, 0) == a(0, r));
         TESTER_ASSERT(row(1, 0) == a(1, r));
         TESTER_ASSERT(row(2, 0) == a(2, r));
         ++r;
      }
      TESTER_ASSERT(r == 2);
   }

   void test_const_columns()
   {
      test_const_columns_impl<Array_row_major<float, 2>>();
      test_const_columns_impl<Array_column_major<float, 2>>();
      test_const_columns_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_const_columns_impl()
   {
      array_type a( 3, 2 );
      a = { 1, 2, 3,
         4, 5, 6 };

      const auto& a_const = a;

      int r = 0;
      for ( auto row : rows( a_const ) )
      {
         TESTER_ASSERT( row.shape() == vector2ui( 3, 1 ) );
         TESTER_ASSERT( row( 0, 0 ) == a( 0, r ) );
         TESTER_ASSERT( row( 1, 0 ) == a( 1, r ) );
         TESTER_ASSERT( row( 2, 0 ) == a( 2, r ) );
         ++r;
      }
      TESTER_ASSERT( r == 2 );
   }
};

TESTER_TEST_SUITE(TestArrayDimIterator);
TESTER_TEST(test_columns);
TESTER_TEST( test_const_columns );
TESTER_TEST_SUITE_END();