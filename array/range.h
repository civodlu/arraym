#pragma once

DECLARE_NAMESPACE_NLL

struct ARRAY_API Range
{
public:
   Range(int minInclusive, int maxInclusive) : min(minInclusive), max(maxInclusive)
   {}

   Range() : min(0), max(0)
   {}

   bool operator==( const Range& other) const
   {
      return other.min == min && other.max == max;
   }

   bool operator!=(const Range& other) const
   {
      return !operator==(other);
   }

   int min;
   int max;
};

inline Range operator+(const Range& range, int value)
{
   NLL_FAST_ASSERT(range.max >= 0 && range.min >= 0, "operate on positive ranges only!");
   return Range(range.min + value, range.max + value);
}

inline Range operator-(const Range& range, int value)
{
   NLL_FAST_ASSERT(range.max >= 0 && range.min >= 0, "operate on positive ranges only!");
   return Range(range.min - value, range.max - value);
}

inline Range operator&(const Range& range1, const Range& range2)
{
   NLL_FAST_ASSERT(range1.max >= 0 && range1.min >= 0, "operate on positive ranges only!");
   NLL_FAST_ASSERT(range2.max >= 0 && range2.min >= 0, "operate on positive ranges only!");
   return Range(std::max(range1.min, range2.min), std::min(range1.max, range2.max));
}

inline Range operator|(const Range& range1, const Range& range2)
{
   NLL_FAST_ASSERT(range1.max >= 0 && range1.min >= 0, "operate on positive ranges only!");
   NLL_FAST_ASSERT(range2.max >= 0 && range2.min >= 0, "operate on positive ranges only!");
   return Range(std::min(range1.min, range2.min), std::max(range1.max, range2.max));
}

static const Range rangeAll = Range(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());

using R = Range; /// short name for range

DECLARE_NAMESPACE_END