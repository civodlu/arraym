#pragma warning(disable : 4503) // decorated names too long

#include "blas-dispatcher.h"

DECLARE_NAMESPACE_NLL

namespace blas
{
details::BlasDispatcherImpl& BlasDispatcher::instance()
{
   static details::BlasDispatcherImpl i;
   return i;
}

namespace details
{

BlasDispatcherImpl::BlasDispatcherImpl(const std::string& path)
{
   readConfiguration(path);
}

void BlasDispatcherImpl::runBenchmark()
{
   // @TODO
}

void BlasDispatcherImpl::readConfiguration(const std::string&)
{
   // @TODO
}

void BlasDispatcherImpl::writeConfiguration(const std::string&) const
{
   // @TODO
}
}
}
DECLARE_NAMESPACE_NLL_END