#ifdef WITH_CUDA

#define CHECK_CUDA(expr) \
   ensure((expr) == cudaError_t::cudaSuccess, "CUBLAS failed!");

template <class T>
class AllocatorCuda : public memory_not_moveable
{
public:
   using value_type = T;
   
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

   T* allocate(std::size_t nb_elements)
   {
      T* gpu_ptr = nullptr;
      CHECK_CUDA(cudaMalloc(&gpu_ptr, nb_elements * sizeof(T)));
      return gpu_ptr;
   }

   void deallocate(T* gpu_ptr, std::size_t n)
   {
      if (gpu_ptr != nullptr)
      {
         CHECK_CUDA(cudaFree(gpu_ptr));
      }
   }
private:
   // stateless
};

#endif