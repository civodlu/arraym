#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;

struct TestContext
{
   struct Context1 : public core::ContextInstance
   {
      std::string value;

      virtual std::shared_ptr<ContextInstance> clone() const override
      {
         ensure( 0, "not implemented!" );
      }
   };

   struct Context2 : public core::ContextInstance
   {
      int value;

      virtual std::shared_ptr<ContextInstance> clone() const override
      {
         ensure( 0, "not implemented!" );
      }
   };

   struct Context3 : public Context2
   {
      int value2;

      virtual std::shared_ptr<ContextInstance> clone() const override
      {
         ensure( 0, "not implemented!" );
      }
   };

   void testAllMethods()
   {
      core::Context context;

      auto c1 = std::make_shared<Context1>();
      c1->value = "test";
      auto c2 = std::make_shared<Context2>();
      c2->value = 42;

      context.add( c1 );
      context.add( c2 );

      auto ptr1 = context.get<Context1>();
      TESTER_ASSERT( ptr1 );
      auto ptr2 = context.get<Context1>();
      TESTER_ASSERT( ptr2 );

      std::shared_ptr<Context1> c1b = context.getSharedPtr<Context1>();
      TESTER_ASSERT( c1b == c1 );

      context.erase( c2.get() );
      TESTER_ASSERT( context.get<Context2>() == nullptr );

      auto c1c = std::make_shared<Context1>();
      c1c->value = "test2";
      context.add( c1c );
      TESTER_ASSERT( context.get<Context1>() );
      TESTER_ASSERT( context.get<Context1>()->value == "test2" );

      std::shared_ptr<Context2> c3( new Context3() );
      context.add( c3 );
      TESTER_ASSERT( context.get<Context2>() == c3.get() );
      TESTER_ASSERT( context.getDerived<Context3>() == c3.get() );

      context.clear();
      TESTER_ASSERT( context.get<Context2>() == 0 );
   }
};

TESTER_TEST_SUITE( TestContext );
TESTER_TEST( testAllMethods );
TESTER_TEST_SUITE_END();