#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

namespace
{
long functionTest(int, std::string&, const std::string&)
{
   return 42;
}

struct Test
{
   long operator()(int, std::string&, const std::string&)
   {
      return 42;
   }
};
}

struct TestArrayTraits
{
   void testFunctionTraits_lambda()
   {
      auto lambda = [](int, std::string&, const std::string&) -> long { return 42; };
      typedef function_traits<decltype(lambda)> traits;
      static_assert(traits::arity == 3, "expected 3 arguments");
      static_assert(std::is_same<long, traits::return_type>::value, "error");
      static_assert(std::is_same<int, traits::argument<0>::type>::value, "error");
      static_assert(std::is_same<std::string&, traits::argument<1>::type>::value, "error");
      static_assert(std::is_same<const std::string&, traits::argument<2>::type>::value, "error");

      static_assert(std::is_same<traits::arguments, std::tuple<int, std::string&, const std::string&>>::value, "error");
   }

   void testFunctionTraits_function()
   {
      typedef function_traits<decltype(functionTest)> traits;
      static_assert(traits::arity == 3, "expected 3 arguments");
      static_assert(std::is_same<long, traits::return_type>::value, "error");
      static_assert(std::is_same<int, traits::argument<0>::type>::value, "error");
      static_assert(std::is_same<std::string&, traits::argument<1>::type>::value, "error");
      static_assert(std::is_same<const std::string&, traits::argument<2>::type>::value, "error");

      static_assert(std::is_same<traits::arguments, std::tuple<int, std::string&, const std::string&>>::value, "error");
   }

   void testFunctionTraits_functor()
   {
      typedef function_traits<Test> traits;
      static_assert(traits::arity == 3, "expected 3 arguments");
      static_assert(std::is_same<long, traits::return_type>::value, "error");
      static_assert(std::is_same<int, traits::argument<0>::type>::value, "error");
      static_assert(std::is_same<std::string&, traits::argument<1>::type>::value, "error");
      static_assert(std::is_same<const std::string&, traits::argument<2>::type>::value, "error");

      static_assert(std::is_same<traits::arguments, std::tuple<int, std::string&, const std::string&>>::value, "error");
   }

   void testFunctionTraits_stdfunction()
   {
      typedef function_traits<std::function<long(int p1, std::string& p2, const std::string& p3)>> traits;
      static_assert(traits::arity == 3, "expected 4 arguments");
      static_assert(std::is_same<long, traits::return_type>::value, "error");
      static_assert(std::is_same<int, traits::argument<0>::type>::value, "error");
      static_assert(std::is_same<std::string&, traits::argument<1>::type>::value, "error");
      static_assert(std::is_same<const std::string&, traits::argument<2>::type>::value, "error");

      static_assert(std::is_same<traits::arguments, std::tuple<int, std::string&, const std::string&>>::value, "error");
   }

   void testFunctionTraits_tupleTail()
   {
      typedef std::tuple<float, const int, double, char> tuple;
      static_assert(std::is_same<tuple_split<tuple>::tail, std::tuple<const int, double, char>>::value, "error");
      static_assert(std::is_same<tuple_split<tuple>::head, float>::value, "error");
   }

   void test_isStaticVector()
   {
      static_assert(is_static_vector<StaticVector<float, 4>>::value, "error!");
      static_assert(!is_static_vector<float>::value, "error!");
   }
};

// these are all static tests
