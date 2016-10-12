#ifndef NLL_CORE_CONTEXT_H_
# define NLL_CORE_CONTEXT_H_

namespace nll
{
namespace core
{
   class INFRA_API ContextInstance : public core::NonCopyable
   {
   public:
      virtual ~ContextInstance()
      {}

      /**
       @brief Clone a context
       */
      virtual std::shared_ptr<ContextInstance> clone() const = 0;
   };

   class INFRA_API Context
   {
      typedef std::map<std::string, std::shared_ptr<ContextInstance>>   ContextContainer;

   public:
      /**
      @brief Stores a context. If one already existed for this type, it previous context is replaced
      */
      template <class T> void add( std::shared_ptr<T>& context )
      {
         static_assert( std::is_base_of<ContextInstance, T>::value, "T must be a derived of ContextInstance" );
         NLL_FAST_ASSERT( context.get(), "must not be empty" );

         const std::string s = typeid( T ).name();
         #ifdef WITH_OMP
         #pragma omp critical
         #endif
         {
            _contexts[ s ] = context;
         }
      }

      /**
      @brief returns the specified context. This must be EXACTLY the same class (and not a inherited one)
      */
      template <class T> T* get() const
      {
         const std::string s = typeid( T ).name();
         ContextContainer::const_iterator it = _contexts.find( s );
         if ( it == _contexts.end() )
            return nullptr;
         return dynamic_cast<T*>( it->second.get() );
      }

      template <class T> std::shared_ptr<T> getSharedPtr() const
      {
         const std::string s = typeid( T ).name();
         ContextContainer::const_iterator it = _contexts.find( s );
         if ( it == _contexts.end() )
            return std::shared_ptr<T>();
         return std::dynamic_pointer_cast<T>( it->second );
      }

      /**
      @brief returns the specified context, this can be any class derived from T
      @note if there are several classes, the first one is returned (this is defined by the type of container)
      */
      template <class T> T* getDerived() const
      {
         for ( ContextContainer::const_iterator it = _contexts.begin(); it != _contexts.end(); ++it )
         {
            T* val = dynamic_cast<T*>( it->second.get() );
            if ( val )
               return val;
         }
         return nullptr;
      }

      /**
      @brief clear all the contexts contained
      */
      void clear()
      {
         _contexts.clear();
      }

      /**
      @brief Remove one particular instance of the context.
      @return true if erased from this context
      */
      template <class T>
      bool erase( T* )
      {
         const std::string s = typeid( T ).name();
         ContextContainer::const_iterator it = _contexts.find( s );
         if ( it != _contexts.end() )
         {
            _contexts.erase( it );
            return true;
         }

         return false;
      }

      size_t size() const
      {
         return _contexts.size();
      }

      std::shared_ptr<Context> clone() const
      {
         auto cpy = std::make_shared<Context>();
         for ( const auto& c : _contexts )
         {
            auto c_cpy = c.second->clone();
            cpy->_contexts[ c.first ] = c_cpy;
         }
         return cpy;
      }

   private:
      ContextContainer     _contexts;
   };
}
}

#endif
