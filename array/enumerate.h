#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
   template <class iterator>
   struct Enumeration
   {
      ui32      index;
      iterator  iterator;

      typename iterator::reference operator*( )
      {
         return *iterator;
      }

      const typename iterator::reference operator*( ) const
      {
         return *iterator;
      }
   };

   template <class iterator>
   struct EnumerateIterator
   {
      EnumerateIterator( iterator current ) : _enumeration( { 0, current } )
      {}

      EnumerateIterator& operator++( )
      {
         ++_enumeration.iterator;
         ++_enumeration.index;
         return *this;
      }

      bool operator==( const EnumerateIterator& it ) const
      {
         return it._enumeration.iterator == _enumeration.iterator;
      }

      bool operator!=( const EnumerateIterator& it ) const
      {
         return !operator==( it );
      }

      Enumeration<iterator>& operator*( )
      {
         return _enumeration;
      }

      const Enumeration<iterator>& operator*( ) const
      {
         return _enumeration;
      }

   private:
      Enumeration<iterator> _enumeration;
   };

   template <class iterator>
   struct EnumerateIteratorProxy
   {
      EnumerateIteratorProxy( EnumerateIterator<iterator> begin, EnumerateIterator<iterator> end ) : _begin( begin ), _end( end )
      {}

      EnumerateIterator<iterator> begin()
      {
         return _begin;
      }

      EnumerateIterator<iterator> end()
      {
         return _end;
      }

   private:
      EnumerateIterator<iterator> _begin;
      EnumerateIterator<iterator> _end;
   };
}

/**
 @brief Create an enumeration on a sequence (i.e., access to the iterator and the index)

 For example,
 std::vector<int> v = {2, 4, 6};
 for (auto& e : v )
 {
   assert(*e == v[e.index]);
 }
 */
template <class Sequence>
details::EnumerateIteratorProxy<typename Sequence::iterator> enumerate( Sequence& sequence )
{
   return details::EnumerateIteratorProxy<typename Sequence::iterator>( std::begin( sequence ), std::end( sequence ) );
}

/**
@brief Create an enumeration on a sequence (i.e., access to the iterator and the index)

For example,
std::vector<int> v = {2, 4, 6};
for (auto& e : v )
{
assert(*e == v[e.index]);
}
*/
template <class Sequence>
details::EnumerateIteratorProxy<typename Sequence::const_iterator> enumerate( const Sequence& sequence )
{
   return details::EnumerateIteratorProxy<typename Sequence::const_iterator>( std::begin( sequence ), std::end( sequence ) );
}

DECLARE_NAMESPACE_NLL_END
