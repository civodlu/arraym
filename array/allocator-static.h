#pragma once

DECLARE_NAMESPACE_NLL

/// tag an allocator to specify when an allocator can't use std::move (e.g., allocator is stack based)
struct memory_not_moveable
{
   // implement bool can_move(T*, size)
};

/**
 @brief Allocator allocating a small static buffer size. If the actual size requested is larger, fall back to another allocator
 @warning Assumed the allocator can only be responsible for at most 1 allocation at any time. Not suitable for generic allocator
 @tparam FallbackAllocator another allocator to fallback in case the memory to be allocated is bigger than the static memory.
         The allocator MUST be stateless
 */
template <class T, size_t StaticNumberOfT, class FallbackAllocator = std::allocator<T>>
class AllocatorSingleStaticMemory : public memory_not_moveable
{
public:
   using value_type         = T;
   using pointer            = T*;
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
   AllocatorSingleStaticMemory(const AllocatorSingleStaticMemory<U, StaticNumberOfT, typename FallbackAllocator::template rebind<U>::other>&)
   {
      // do not copy state
   }

   T* allocate(std::size_t n)
   {
      if (n > StaticNumberOfT)
      {
         fallback_allocator a;
         _allocated = std::allocator_traits<fallback_allocator>::allocate(a, n);
         return _allocated;
      }
      else
      {
         return _staticMemory;
      }
   }

   void deallocate(T* p, std::size_t n)
   {
      if (n > StaticNumberOfT)
      {
         fallback_allocator a;
         std::allocator_traits<fallback_allocator>::deallocate(a, p, n);
      }
   }

   /**
    @brief return if a pointer can be shared with a different allocator.

    E.g., size < StaticNumberOfT, we have memory allocated on the stack so we can't move it
    */
   static bool can_move(T*, size_t size)
   {
      return size > StaticNumberOfT;
   }

private:
   T _staticMemory[StaticNumberOfT];
   T* _allocated = nullptr;
};

DECLARE_NAMESPACE_NLL_END
