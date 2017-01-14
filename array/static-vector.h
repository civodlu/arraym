#pragma once

DECLARE_NAMESPACE_NLL

struct No_init_tag
{
};

const static No_init_tag no_init_tag;

/**
@ingroup core
@brief implement static vectors. Memory is allocated on the heap. Memory is not shared accross instances.
*/
template <class T, size_t SIZE>
class StaticVector
{
public:
   typedef T value_type;
   typedef T* iterator;
   typedef const T* const_iterator;

   typedef T* pointer;
   typedef const T* const_pointer;
   typedef T& reference;
   typedef const T& const_reference;

   enum
   {
      sizeDef = SIZE
   };

   static_assert(std::is_trivially_copyable<T>::value, "must be trivial to copy");

public:
   /**
   @brief copy constructor
   */
   StaticVector(const StaticVector& cpy)
   {
      std::memcpy(begin(), cpy.begin(), SIZE * sizeof(T));
   }

   /**
   @brief instantiate a vector
   */
   StaticVector()
   {
      std::fill_n(begin(), SIZE, T());
   }

   explicit StaticVector(const No_init_tag&)
   {
      // do nothing
   }

   template <class T2>
   explicit StaticVector(const StaticVector<T2, SIZE>& cpy)
   {
      for (size_t n = 0; n < SIZE; ++n)
      {
         begin()[n] = static_cast<T>(cpy[n]);
      }
   }

   template <typename T2, typename... Args, typename = typename std::enable_if<sizeof...(Args) + 1 == SIZE>::type>
   StaticVector(const T2& arg1, const Args&... args) // avoid "shadowing" the default constructor with an initia argument
   {
      begin()[0] = static_cast<T>(arg1);
      init<1>(args...);
   }

   explicit StaticVector(const T& value)
   {
      std::fill_n(begin(), SIZE, value);
   }

private:
   template <size_t Dim, typename T2, typename... Args>
   FORCE_INLINE void init(const T2& value, const Args&... args)
   {
      begin()[Dim] = static_cast<T>(value);
      init<Dim + 1>(args...);
   }

   // end recursion
   template <size_t Dim>
   FORCE_INLINE void init()
   {
   }

public:
   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE const T& at(const size_t index) const
   {
      NLL_FAST_ASSERT(index < SIZE, "out of bound index");
      return _buffer[index];
   }

   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE T& at(const size_t index)
   {
      NLL_FAST_ASSERT(index < SIZE, "out of bound index");
      return _buffer[index];
   }

   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE const T& operator[](const size_t index) const
   {
      return at(index);
   }

   /**
   @brief return the value at the specified index
   */
   template <size_t N>
   FORCE_INLINE const T& get() const
   {
      static_assert(N < SIZE, "N must be < SIZE");
      return _buffer[N];
   }

   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE T& operator[](const size_t index)
   {
      return at(index);
   }

   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE const T& operator()(const size_t index) const
   {
      return at(index);
   }

   /**
   @brief return the value at the specified index
   */
   FORCE_INLINE T& operator()(const size_t index)
   {
      return at(index);
   }

   template <class T2>
   StaticVector<T2, SIZE> staticCastTo() const
   {
      StaticVector<T2, SIZE> casted;
      for (size_t n = 0; n < SIZE; ++n)
      {
         casted[n] = static_cast<T2>(_buffer[n]);
      }
      return casted;
   }

   FORCE_INLINE static size_t size()
   {
      return SIZE;
   }

   /**
   @brief print the vector to a stream
   */
   void print(std::ostream& o) const
   {
      o << "[";
      if (SIZE)
      {
         for (size_t n = 0; n + 1 < SIZE; ++n)
            o << at(n) << " ";
         o << at(SIZE - 1);
      }
      o << "]";
   }

   void write(std::ostream& f) const
   {
      f.write((char*)_buffer, sizeof(T) * SIZE);
   }

   void read(std::istream& f)
   {
      f.read((char*)_buffer, sizeof(T) * SIZE);
   }

   FORCE_INLINE bool equal(const StaticVector& r, T eps = (T)1e-5) const
   {
      for (size_t n = 0; n < SIZE; ++n)
      {
         if (!NAMESPACE_NLL::equal<T>(_buffer[n], r[n], eps))
            return false;
      }
      return true;
   }

   FORCE_INLINE StaticVector<T, SIZE>& operator=(T value)
   {
      for (size_t n = 0; n < SIZE; ++n)
      {
         _buffer[n] = value;
      }
      return *this;
   }

   FORCE_INLINE bool operator==(const StaticVector& r) const
   {
      return equal(r);
   }

   FORCE_INLINE bool operator!=(const StaticVector& r) const
   {
      return !(*this == r);
   }

   FORCE_INLINE iterator begin()
   {
      return _buffer;
   }

   FORCE_INLINE const_iterator begin() const
   {
      return _buffer;
   }

   FORCE_INLINE iterator end()
   {
      return _buffer + SIZE;
   }

   FORCE_INLINE const_iterator end() const
   {
      return _buffer + SIZE;
   }

protected:
   T _buffer[SIZE];
};

/**
@ingroup core
@brief equal operator
*/
template <class T, size_t SIZE>
bool operator==(const StaticVector<T, SIZE>& l, const StaticVector<T, SIZE>& r)
{
   for (size_t n = 0; n < SIZE; ++n)
   {
      if (!equal(l[n], r[n]))
         return false;
   }
   return true;
}

template <class T, size_t Size>
bool equal(const StaticVector<T, Size>& lhs, const StaticVector<T, Size>& rhs, T eps = (T)1e-5)
{
   return lhs.equal(rhs, eps);
}

template <class T, size_t SIZE>
std::ostream& operator<<(std::ostream& o, const StaticVector<T, SIZE>& v)
{
   v.print(o);
   return o;
}

DECLARE_NAMESPACE_NLL_END