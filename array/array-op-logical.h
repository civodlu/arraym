#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
   template <class T1, class T2, class T3, class F>
   void apply_fun2_array_strided( T1* output, ui32 output_stride, const T2* input1, ui32 input1_stride, const T3* input2, ui32 input2_stride, ui32 nb_elements, F f )
   {
      static_assert( is_callable_with<F, T2, T3>::value, "Op is not callable with the correct arguments!" );

      const T1* output_end = output + output_stride * nb_elements;
      for ( ; output != output_end; output += output_stride, input1 += input1_stride, input2 += input2_stride )
      {
         *output = f( *input1, *input2 );
      }
   };

   template <class Output>
   struct ApplyBinOp
   {
      template <class T1, class T2, class Config1, class Config2, size_t N, class Op>
      typename Array<T1, N, Config1>::template rebind<Output>::other operator()( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2, Op& op )
      {
         NLL_FAST_ASSERT( a1.shape() == a2.shape(), "must have the same shape!" );
         using result_type = typename Array<T1, N, Config1>::template rebind<ui8>::other;

         auto f = [&]( ui8* output, ui32 output_stride, const T1* input1, ui32 input1_stride, const T2* input2, ui32 input2_stride, ui32 nb_elements )
         {
            details::apply_fun2_array_strided( output, output_stride, input1, input1_stride, input2, input2_stride, nb_elements, op );
         };

         result_type result( a1.shape() );
         iterate_array_constarray_constarray( result, a1, a2, f );
         return result;
      }
   };

   template <class Output>
   struct ApplyUnaryOp
   {
      template <class T1, class Config1, size_t N, class Op>
      typename Array<T1, N, Config1>::template rebind<Output>::other operator()( const Array<T1, N, Config1>& a1, Op& op )
      {
         using result_type = typename Array<T1, N, Config1>::template rebind<ui8>::other;

         auto f = [&]( Output* output, ui32 output_stride, const T1* input1, ui32 input1_stride, ui32 nb_elements )
         {
            details::apply_fun_array_strided( output, output_stride, input1, input1_stride, nb_elements, op );
         };

         result_type result( a1.shape() );
         iterate_array_constarray( result, a1, f );
         return result;
      }
   };
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::less<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<( const Array<T1, N, Config1>& a1, T2 value )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v < value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<=( const Array<T1, N, Config1>& a1, T2 value )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v <= value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<( T2 value, const Array<T1, N, Config1>& a1 )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v > value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<=( T2 value, const Array<T1, N, Config1>& a1 )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v >= value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator<=( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::less_equal<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::greater<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>=( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::greater_equal<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>( const Array<T1, N, Config1>& a1, T2 value )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v > value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>=( const Array<T1, N, Config1>& a1, T2 value )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v >= value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>( T2 value, const Array<T1, N, Config1>& a1 )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v < value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class Config1, size_t N, class T2, typename = typename std::enable_if<std::is_convertible<T2, T1>::value>::type>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator>=( T2 value, const Array<T1, N, Config1>& a1 )
{
   auto value_converted = static_cast<T1>( value );

   auto op = [&]( T1 v )->ui8
   {
      return v <= value_converted;
   };
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator&( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::logical_and<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator|( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::logical_or<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other equal_elementwise( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::equal_to<>();
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class T2, class Config1, class Config2, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other different_elementwise( const Array<T1, N, Config1>& a1, const Array<T2, N, Config2>& a2 )
{
   auto op = std::not_equal_to<>;
   return details::ApplyBinOp<ui8>()( a1, a2, op );
}

template <class T1, class Config1, size_t N>
typename Array<T1, N, Config1>::template rebind<ui8>::other operator!( const Array<T1, N, Config1>& a1 )
{
   std::logical_not<> op;
   return details::ApplyUnaryOp<ui8>()( a1, op );
}

/**
@brief Return true if any predicate is true

any([1, 2, 3, 4, 5], value > 4) = true
any([1, 2, 3, 4, 5], value > 14) = false

if indexes_out != nullptr, the full list of indexes satisfying the predicate will be returned (i.e., no early stopping)

if index_out != nullptr, one index satisfying the predicate will be returned (i.e., no early stopping)
*/
template <class T, size_t N, class Config, class Predicate>
bool any(const Array<T, N, Config>& array, Predicate p, StaticVector<ui32, N>* index_out = nullptr, std::vector<StaticVector<ui32, N>>* indexes_out = nullptr)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;
   StaticVector<ui32, N> index;

   ConstArrayProcessor_contiguous_byMemoryLocality<array_type> processor(array, 0);
   bool hasMoreElements = true;
   ui32 current_varying_index;

   bool result = false;
   auto functor = [&](T value)
   {
      if (p(value))
      {
         result = true;
         if (index_out)
         {
            index = processor.getArrayIndex();
            index[processor.getVaryingIndex()] += current_varying_index;
            *index_out = index;
            index_out = nullptr;
         }

         if (indexes_out)
         {
            index = processor.getArrayIndex();
            index[processor.getVaryingIndex()] += current_varying_index;
            indexes_out->push_back(index);
         }
         
         if (!indexes_out)
         {
            processor.stop();  // early stop, we know the result is true
         }
      }
   };

   while (hasMoreElements)
   {
      const_pointer_type ptr(nullptr);
      hasMoreElements = processor.accessMaxElements(ptr);

      const auto y_stride = processor.stride();
      const T* y_end = ptr + y_stride * processor.getNbElementsPerAccess();
      for (current_varying_index = 0; ptr != y_end; ptr += y_stride, ++current_varying_index)
      {
         functor(*ptr);
      }
   }
   return result;
}

/**
@brief Return true if all predicate is true for all values

all([1, 2, 3, 4, 5], value > 0) = true
all([1, 2, 3, 4, 5], value > 4) = false
*/
template <class T, size_t N, class Config, class Predicate>
bool all(const Array<T, N, Config>& array, Predicate p)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;

   ConstArrayProcessor_contiguous_byMemoryLocality<array_type> processor(array, 0);
   bool hasMoreElements = true;

   bool result = true;
   auto functor = [&](T value)
   {
      if (!p(value))
      {
         result = false;
         processor.stop();  // early stop, we know the result is true
      }
   };

   while (hasMoreElements)
   {
      const_pointer_type ptr(nullptr);
      hasMoreElements = processor.accessMaxElements(ptr);

      details::apply_naive1_const(ptr, processor.stride(), processor.getNbElementsPerAccess(), functor);
   }
   return result;
}

DECLARE_NAMESPACE_NLL_END
