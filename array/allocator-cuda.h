#ifdef WITH_CUDA

#define CHECK_CUDA(expr) \
   ensure((expr) == cudaError_t::cudaSuccess, "CUBLAS failed!");

#endif

DECLARE_NAMESPACE_NLL

#ifdef WITH_CUDA

template <class T>
class AllocatorCuda
{
public:
   using value_type = T;
   using pointer = cuda_ptr<T>;
   
   template <class U>
   struct rebind
   {
      using other = AllocatorCuda<U>;
   };

   AllocatorCuda()
   {
      // do not copy state
   }

   template <class U>
   AllocatorCuda(const AllocatorCuda<U>&)
   {
      // do not copy state
   }

   pointer allocate(std::size_t nb_elements)
   {
      T* gpu_ptr = nullptr;
      if (nb_elements)
      {
         CHECK_CUDA(cudaMalloc(&gpu_ptr, nb_elements * sizeof(T)));
      }
      return pointer(gpu_ptr);
   }

   void deallocate(pointer gpu_ptr, std::size_t n)
   {
      if (n)
      {
         // first GPU memory block can be 0
         CHECK_CUDA(cudaFree(gpu_ptr));
      }
   }

   void construct_n(size_t nb_elements, pointer gpu_ptr, T default_value)
   {
      // just initialize with default value
      cuda::kernel_init(gpu_ptr, nb_elements, default_value);
   }

   void destroy_n(pointer UNUSED(gpu_ptr), size_t UNUSED(nb_elements))
   {
      // do nothing: GPU won't store complex data...
   }

private:
   // stateless
};

#endif

template <class Allocator>
struct is_allocator_gpu : public std::false_type
{};

#ifdef WITH_CUDA
template <class T>
struct is_allocator_gpu<AllocatorCuda<T>> : public std::true_type
{};
#endif

template <class array_type>
struct array_remove_const
{
   using type = array_type;
};

template <class T>
struct array_remove_const<const T*>
{
   using type = T*;
};

#ifdef WITH_CUDA
template <class T>
struct array_remove_const<cuda_ptr<T>>
{
   using type = cuda_ptr<T>;
};

template <class T>
struct array_remove_const<const cuda_ptr<T>>
{
   using type = cuda_ptr<T>;
};
#endif

template <class T>
struct value_type_inf
{
   using type = T;
};

template <class T>
struct value_type_inf<T*>
{
   using type = typename value_type_inf<T>::type;
};

template <class array_type>
struct array_add_const
{
   using value_type = typename value_type_inf<array_type>::type;
   using type = const value_type*;
};

#ifdef WITH_CUDA
template <class T>
struct array_add_const<cuda_ptr<T>>
{
   using type = const cuda_ptr<T>;
};

template <class T>
struct array_add_const<const cuda_ptr<T>>
{
   using type = const cuda_ptr<T>;
};
#endif

DECLARE_NAMESPACE_NLL_END
