#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief return the inverse of the input
@note Do no throw if the matrix is singular. Instead, the returned size if (0,0)
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> inv_nothrow(const Array<T, 2, Config>& a)
{
   using matrix_type = Array<T, 2, Config>;
   ensure(a.rows() == a.columns(), "must be square!");
   matrix_type i = a;

   using allocator_type = typename matrix_type::allocator_type::template rebind<blas::BlasInt>::other;
   using pointer        = typename allocator_type::pointer;
   const auto size      = static_cast<blas::BlasInt>(a.rows());
   allocator_type allocator_int;
   auto IPIV_ptr = std::allocator_traits<allocator_type>::allocate(allocator_int, size + 1);

   auto deleter = [&](int* ptr) { allocator_int.deallocate(pointer(ptr), size + 1); }; // we MUST reuse the proper deallocator (e.g., CUDA based allocators)
   std::unique_ptr<blas::BlasInt, decltype(deleter)> IPIV(IPIV_ptr, deleter);

   const auto memory_order = getMatrixMemoryOrder(a);
   const auto lda          = leading_dimension<T, Config>(i);

   const auto r = blas::getrf<T>(memory_order, size, size, array_base_memory(i), lda, IPIV_ptr);
   if (r != 0)
   {
      // something is wrong... just return an empty array
      return matrix_type();
   }

   const auto r2 = blas::getri<T>(memory_order, size, array_base_memory(i), lda, IPIV_ptr);
   if (r2 != 0)
   {
      // something is wrong... just return an empty array
      return matrix_type();
   }

   return i;
}

/**
@brief return the inverse of the input

Throw an exception if the inverse failed
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> inv(const Array<T, 2, Config>& a)
{
   auto r = inv_nothrow(a);
   ensure(r.size() != 0, "inv failed!");
   return r;
}

DECLARE_NAMESPACE_NLL_END
