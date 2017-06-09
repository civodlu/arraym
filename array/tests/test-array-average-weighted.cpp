#include <core/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayBlend
{
   void testArrayBlend_float()
   {
      using array_type = Array<float, 2>;

      array_type a1(3, 2);
      a1 = { 1, 2, 3,
             4, 5, 6 };
      
      array_type a2(3, 2);
      a2 = { 11, 12, 13,
             14, 15, 16 };

      auto r = average_weighted<float>(make_stdvector<const array_type*>(&a1, &a2), make_stdvector<float>(1.0f, 3.0f));
      TESTER_ASSERT(r.shape() == a1.shape());

      const auto error = norm2(r - (a1 * 0.25 + a2 * 0.75));
      TESTER_ASSERT(error < 1e-5f);
   }

   void testArrayBlend_vec()
   {
      using array_type = Array<vector2f, 2>;

      array_type a1(3, 2);
      a1 = { vector2f(1, 2), vector2f(3, 4), vector2f(5, 6),
             vector2f(7, 8), vector2f(9, 10), vector2f(11, 12) };

      array_type a2(3, 2);
      a2 = { vector2f(11, 12), vector2f(13, 14), vector2f(15, 16),
             vector2f(17, 18), vector2f(19, 110), vector2f(111, 112) };

      auto r = average_weighted<vector2f>(make_stdvector<const array_type*>(&a1, &a2), make_stdvector<float>(1.0f, 3.0f));

      TESTER_ASSERT(r.shape() == a1.shape());

      TESTER_ASSERT(r(0, 0) == a1(0, 0) * 0.25f + a2(0, 0) * 0.75f);
      TESTER_ASSERT(r(1, 0) == a1(1, 0) * 0.25f + a2(1, 0) * 0.75f);
      TESTER_ASSERT(r(2, 0) == a1(2, 0) * 0.25f + a2(2, 0) * 0.75f);

      TESTER_ASSERT(r(0, 1) == a1(0, 1) * 0.25f + a2(0, 1) * 0.75f);
      TESTER_ASSERT(r(1, 1) == a1(1, 1) * 0.25f + a2(1, 1) * 0.75f);
      TESTER_ASSERT(r(2, 1) == a1(2, 1) * 0.25f + a2(2, 1) * 0.75f);
   }

   void testArrayBlend_char()
   {
      using array_type = Array<ui8, 2>;

      array_type a1(3, 2);
      a1 = { 1, 2, 3,
         4, 255, 255 };

      array_type a2(3, 2);
      a2 = { 11, 12, 13,
         64, 15, 255 };

      auto r = average_weighted<ui8>(make_stdvector<const array_type*>(&a1, &a2), make_stdvector<float>(1.0f, 3.0f));
      TESTER_ASSERT(r.shape() == a1.shape());

      std::cout << r.cast<float>() << std::endl;

      const auto max_value = max(abs(r.cast<float>() - (a1.cast<float>() * 0.25 + a2.cast<float>() * 0.75)));
      std::cout << "max=" << (float)max_value << std::endl;
      TESTER_ASSERT(max_value < 2.0f);
   }
};

TESTER_TEST_SUITE(TestArrayBlend);
TESTER_TEST(testArrayBlend_float);
TESTER_TEST(testArrayBlend_vec);
TESTER_TEST(testArrayBlend_char);
TESTER_TEST_SUITE_END();
