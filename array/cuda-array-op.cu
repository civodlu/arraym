#include "forward.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

DECLARE_NAMESPACE_NLL

namespace cuda
{
   /**
   @brief strided range
   see https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
   */
   template <typename Iterator>
   class strided_range
   {
   public:
      typedef typename thrust::iterator_difference<Iterator>::type difference_type;

      struct stride_functor : public thrust::unary_function<difference_type, difference_type>
      {
         difference_type stride;

         stride_functor(difference_type stride)
            : stride(stride) {}

         __host__ __device__
            difference_type operator()(const difference_type& i) const
         {
            return stride * i;
         }
      };

      typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
      typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
      typedef typename thrust::permutation_iterator<Iterator, TransformIterator>    PermutationIterator;

      // type of the strided_range iterator
      typedef PermutationIterator iterator;

      // construct strided_range for the range [first,last)
      strided_range(Iterator first, Iterator last, difference_type stride)
         : first(first), last(last), stride(stride) {}

      iterator begin() const
      {
         return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
      }

      iterator end() const
      {
         return begin() + ((last - first) + (stride - 1)) / stride;
      }

   protected:
      Iterator first;
      Iterator last;
      difference_type stride;
   };
}

namespace details
{
   template <class T, class F>
   void apply_fun_cuda(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements, F f)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)(input));
      thrust::device_ptr<T> dev_ptr_out((T*)output);

      using strided_range = cuda::strided_range<thrust::device_ptr<T>>;
      auto range_input = strided_range(dev_ptr_in, dev_ptr_in + nb_elements * input_stride, input_stride);
      auto range_output = strided_range(dev_ptr_out, dev_ptr_out + nb_elements * output_stride, output_stride);
      thrust::transform(range_input.begin(), range_input.end(), range_output.begin(), f);
   }

   template <class T>
   void cos(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){ return std::cos(value); });
   }

   template <class T>
   void sin(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return std::sin(value);
      });
   }

   template <class T>
   void sqrt(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return std::sqrt(value);
      });
   }

   template <class T>
   void sqr(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return value * value;
      });
   }

   template <class T>
   void abs(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return std::abs(value);
      });
   }

   template <class T>
   void exp(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return std::exp(value);
      });
   }

   template <class T>
   void log(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      apply_fun_cuda(output, output_stride, input, input_stride, nb_elements,
         []__device__(T value){
         return std::log(value);
      });
   }

   template <class T>
   T max(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)input);
      using strided_range = cuda::strided_range<thrust::device_ptr<T>>;
      auto range_input = strided_range(dev_ptr_in, dev_ptr_in + nb_elements * input_stride, input_stride);

      return *thrust::max_element(range_input.begin(), range_input.end());
   }

   template <class T>
   T min(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)input);
      using strided_range = cuda::strided_range<thrust::device_ptr<T>>;
      auto range_input = strided_range(dev_ptr_in, dev_ptr_in + nb_elements * input_stride, input_stride);

      return *thrust::min_element(range_input.begin(), range_input.end());
   }

   template <class T, class Accum>
   Accum sum(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)input);
      using strided_range = cuda::strided_range<thrust::device_ptr<T>>;
      auto range_input = strided_range(dev_ptr_in, dev_ptr_in + nb_elements * input_stride, input_stride);

      return thrust::reduce(range_input.begin(), range_input.end());
   }

   template ARRAY_API void cos(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void sin(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void log(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void exp(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void sqr(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void sqrt(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API void abs(cuda_ptr<float> output, ui32 output_stride, const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);

   template ARRAY_API float max(const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API float min(const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
   template ARRAY_API float sum(const cuda_ptr<float> input, ui32 input_stride, ui32 nb_elements);
}

DECLARE_NAMESPACE_NLL_END