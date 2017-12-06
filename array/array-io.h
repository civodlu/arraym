#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
   namespace
   {
      static const ui32 ARRAY_IO_MAGIC = 0x1234ab91;
   }

   /**
    @brief Write the data protion only of the buffer

    @warning the storage order is depend on the array data layout
    */
   template <class T, size_t N, class Config>
   void write_data(const Array<T, N, Config>& array, std::ostream& f)
   {
      // finally write the data. Carefull: this is not independent of array ordering.
      // if we write with row-major and read with column-major, we need to transpose the data.
      // Purpose is to maximize the IO speed
      using array_type = Array<T, N, Config>;
      using const_pointer_type = typename array_type::const_pointer_type;

      auto op_write = [&](const_pointer_type ptr, ui32 stride, ui32 nb_elements)
      {
         write_naive(ptr, stride, nb_elements, f);
      };

      iterate_constarray(array, op_write);
      ensure(f.good(), "array stream failure!"); // finally make sure there was no error during IO operations
   }

   /**
    @brief write an array to a binary stream
    */
   template <class T, size_t N, class Config>
   void write(const Array<T, N, Config>& array, std::ostream& f)
   {
      const ui32 magic = ARRAY_IO_MAGIC;
      const ui32 nb_dims = N;
      const ui32 bytes_per_element = sizeof(T);

      // write the header
      f.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
      f.write(reinterpret_cast<const char*>(&nb_dims), sizeof(nb_dims));
      f.write(reinterpret_cast<const char*>(&bytes_per_element), sizeof(bytes_per_element));
      f.write(reinterpret_cast<const char*>(array.shape().begin()), sizeof(ui32) * nb_dims);

      if (array.size())
      {
         write_data(array, f);
      }
   }

   /**
   @brief Read the data protion only of the buffer

   Assumed the array is already allocated with the correct shape

   @warning the storage order is depend on the array data layout
   */
   template <class T, size_t N, class Config>
   void read_data(Array<T, N, Config>& array, std::istream& f)
   {
      // finally read the data. Carefull: this is not independent of array ordering.
      using array_type = Array<T, N, Config>;
      using pointer_type = typename array_type::pointer_type;

      auto op_read = [&](pointer_type ptr, ui32 stride, ui32 nb_elements)
      {
         read_naive(ptr, stride, nb_elements, f);
      };

      iterate_array(array, op_read);
      ensure(f.good(), "array stream failure!"); // finally make sure there was no error during IO operations
   }

   /**
   @brief read an array from a binary stream
   */
   template <class T, size_t N, class Config>
   void read(Array<T, N, Config>& array, std::istream& f)
   {
      ui32 magic = 0;
      ui32 nb_dims = 0;
      ui32 bytes_per_element = 0;
      StaticVector<ui32, N> shape;

      // read the header
      f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
      f.read(reinterpret_cast<char*>(&nb_dims), sizeof(nb_dims));
      f.read(reinterpret_cast<char*>(&bytes_per_element), sizeof(bytes_per_element));
      f.read(reinterpret_cast<char*>(shape.begin()), sizeof(ui32) * nb_dims);

      ensure(magic == ARRAY_IO_MAGIC, "magic number is wrong: this data is not an Array data!");
      ensure(nb_dims == N, "number of dimensions mismatch. Expected=" << nb_dims);
      ensure(bytes_per_element == sizeof(T), "element size mismatch. Expected=" << bytes_per_element);

      if (shape != array.shape())
      {
         // resize the array
         array = Array<T, N, Config>(shape);
      }

      if (shape[0])
      {
         read_data(array, f);
      }
   }
}

DECLARE_NAMESPACE_NLL_END