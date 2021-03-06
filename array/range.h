#pragma once

DECLARE_NAMESPACE_NLL

/**
Copyright 2012�2016 Konrad Rudolph

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/**
This version was modified from https://github.com/klmr/cpp11-range

In particular:
- infinite ranges were removed to be more similar to python ranges
  where range(10) = range(0, 10)
- no 'indices' instead just function range over collections, initializer list, arrays

Additional operators were also added for ease of use.
*/

namespace detail
{
template <typename T>
struct range_iter_base : std::iterator<std::input_iterator_tag, T>
{
   DEVICE_CALLABLE
   range_iter_base(T current) : current(current)
   {
   }

   DEVICE_CALLABLE
   T operator*() const
   {
      return current;
   }

   DEVICE_CALLABLE
   T const* operator->() const
   {
      return &current;
   }

   DEVICE_CALLABLE
   range_iter_base& operator++()
   {
      ++current;
      return *this;
   }

   DEVICE_CALLABLE
   range_iter_base operator++(int)
   {
      auto copy = *this;
      ++*this;
      return copy;
   }

   DEVICE_CALLABLE
   bool operator==(range_iter_base const& other) const
   {
      return current == other.current;
   }

   DEVICE_CALLABLE
   bool operator!=(range_iter_base const& other) const
   {
      return !(*this == other);
   }

protected:
   T current;
};

} // namespace detail

template <typename T>
struct range_proxy
{
   struct iter : detail::range_iter_base<T>
   {
      DEVICE_CALLABLE
      iter(T current) : detail::range_iter_base<T>(current)
      {
      }
   };

   struct step_range_proxy
   {
      struct iter : detail::range_iter_base<T>
      {
         DEVICE_CALLABLE
         iter(T current, T step) : detail::range_iter_base<T>(current), step(step)
         {
         }

         using detail::range_iter_base<T>::current;

         DEVICE_CALLABLE
         iter& operator++()
         {
            current += step;
            return *this;
         }

         DEVICE_CALLABLE
         iter operator++(int)
         {
            auto copy = *this;
            ++*this;
            return copy;
         }

         // Loses commutativity. Iterator-based ranges are simply broken. :-(
         DEVICE_CALLABLE
         bool operator==(iter const& other) const
         {
            return step > 0 ? current >= other.current : current < other.current;
         }

         DEVICE_CALLABLE
         bool operator!=(iter const& other) const
         {
            return !(*this == other);
         }

         T step;
      };

      DEVICE_CALLABLE
      step_range_proxy(T begin, T end, T step) : begin_(begin, step), end_(end, step)
      {
      }

      DEVICE_CALLABLE
      iter begin() const
      {
         return begin_;
      }

      DEVICE_CALLABLE
      iter end() const
      {
         return end_;
      }

   private:
      iter begin_;
      iter end_;
   };

   DEVICE_CALLABLE
   range_proxy(T begin, T end) : begin_(begin), end_(end)
   {
   }

   DEVICE_CALLABLE
   step_range_proxy step(T step)
   {
      return {*begin_, *end_, step};
   }

   DEVICE_CALLABLE
   iter begin() const
   {
      return begin_;
   }

   DEVICE_CALLABLE
   iter end() const
   {
      return end_;
   }

   bool operator==(range_proxy other) const
   {
      return begin_ == other.begin_ && end_ == other.end_;
   }

   bool operator!=(range_proxy other) const
   {
      return !operator==(other);
   }

private:
   iter begin_;
   iter end_;
};

namespace traits
{
template <typename C>
struct has_size
{
   template <typename T>
   static auto check(T*) -> typename std::is_integral<decltype(std::declval<T const>().size())>::type;

   template <typename>
   static auto check(...) -> std::false_type;

   using type              = decltype(check<C>(0));
   static const bool value = type::value;
};
}

template <typename T>
DEVICE_CALLABLE range_proxy<T> range(T begin, T end)
{
   return {begin, end};
}

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
DEVICE_CALLABLE range_proxy<T> range(T end)
{
   return {0, end};
}

template <class T>
inline range_proxy<T> operator+(const range_proxy<T>& range, T value)
{
   NLL_FAST_ASSERT(*range.end() >= 0 && *range.begin() >= 0, "operate on positive ranges only!");
   return range_proxy<T>(*range.begin() + value, *range.end() + value);
}

template <class T>
inline range_proxy<T> operator-(const range_proxy<T>& range, T value)
{
   NLL_FAST_ASSERT(*range.end() >= 0 && *range.begin() >= 0, "operate on positive ranges only!");
   return range_proxy<T>(*range.begin() - value, *range.end() - value);
}

template <class T>
inline range_proxy<T> operator&(const range_proxy<T>& range1, const range_proxy<T>& range2)
{
   NLL_FAST_ASSERT(*range1.end() >= 0 && *range1.begin() >= 0, "operate on positive ranges only!");
   NLL_FAST_ASSERT(*range2.end() >= 0 && *range2.begin() >= 0, "operate on positive ranges only!");
   return range_proxy<T>(std::max(*range1.begin(), *range2.begin()), std::min(*range1.end(), *range2.end()));
}

template <class T>
inline range_proxy<T> operator|(const range_proxy<T>& range1, const range_proxy<T>& range2)
{
   NLL_FAST_ASSERT(*range1.end() >= 0 && *range1.begin() >= 0, "operate on positive ranges only!");
   NLL_FAST_ASSERT(*range2.end() >= 0 && *range2.begin() >= 0, "operate on positive ranges only!");
   return range_proxy<T>(std::min(*range1.begin(), *range2.begin()), std::max(*range1.end(), *range2.end()));
}

template <typename C, typename = typename std::enable_if<traits::has_size<C>::value>::type>
DEVICE_CALLABLE auto range(C const& cont) -> range_proxy<decltype(cont.size())>
{
   return {0, cont.size()};
}

template <typename T, std::size_t N>
DEVICE_CALLABLE range_proxy<std::size_t> range(T (&)[N])
{
   return {0, N};
}

template <typename T>
range_proxy<typename std::initializer_list<T>::size_type> DEVICE_CALLABLE range(std::initializer_list<T>&& cont)
{
   return {0, cont.size()};
}

template <class T>
struct is_range : public std::false_type
{
};

template <class T>
struct is_range<range_proxy<T>> : public std::true_type
{
};

template <typename... Args>
struct is_range_n;

template <>
struct is_range_n<> : public std::true_type
{
};

template <class Arg, typename... Args>
struct is_range_n<Arg, Args...>
{
   static const bool value = is_range<typename remove_cvr<Arg>::type>::value && is_range_n<Args...>::value;
};

static const range_proxy<int> rangeAll = range_proxy<int>(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
using R                                = range_proxy<int>; /// short name for range

DECLARE_NAMESPACE_NLL_END
