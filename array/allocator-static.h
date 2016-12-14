#pragma once

DECLARE_NAMESPACE_NLL

/**
 @brief Allocator allocating a small static buffer size. If the actual size requested is larger, fall back to another allocator

 @tparam FallbackAllocator another allocator to fallback in case the memory to be allocated is bigger than the static memory.
         The allocator MUST be stateless
 */
template <class T, size_t StaticNumberOfT, class FallbackAllocator = std::allocator<T>>
class AllocatorSingleStaticMemory
{
public:
   using value_type = T;
   using fallback_allocator = FallbackAllocator;

   template <class U>
   struct rebind
   {
      using other = AllocatorSingleStaticMemory<U, StaticNumberOfT, typename fallback_allocator::template rebind<U>::other>;
   };

   AllocatorSingleStaticMemory()
   {
      // do not copy state
   }

   template <class U>
   AllocatorSingleStaticMemory( const AllocatorSingleStaticMemory<U, StaticNumberOfT, typename FallbackAllocator::template rebind<U>::other>& )
   {
      // do not copy state
   }

   T* allocate( std::size_t n )
   {
      if ( n > StaticNumberOfT )
      {
         fallback_allocator a;
         _allocated = std::allocator_traits<fallback_allocator>::allocate( a, n );
         return _allocated;
      } else
      {
         return _staticMemory;
      }
   }

   void deallocate( T* p, std::size_t n )
   {
      if ( n > StaticNumberOfT )
      {
         fallback_allocator a;
         std::allocator_traits<fallback_allocator>::deallocate( a, p, n );
      }
   }

private:
   T  _staticMemory[ StaticNumberOfT ];
   T* _allocated = nullptr;
};

/*
// return that all specializations of this allocator are interchangeable
template <class T, class U, size_t StaticNumberOfT, class FallbackAllocator>
bool operator==( const AllocatorSingleStaticMemory<T, StaticNumberOfT, FallbackAllocator>&, const AllocatorSingleStaticMemory<U, StaticNumberOfT, FallbackAllocator>& )
{
   return true;
}

template <class T, class U, size_t StaticNumberOfT, class FallbackAllocator>
bool operator!=( const AllocatorSingleStaticMemory<T>&, const AllocatorSingleStaticMemory<U>& )
{
   return false;
}
*/

DECLARE_NAMESPACE_END