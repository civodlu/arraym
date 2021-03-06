#pragma once

#pragma warning(                                                                                                                                               \
    disable : 4099) // warning LNK4099: PDB 'vc120.pdb' was not found with 'blasd.lib(saxpy.obj)' or at 'C:\devel\nll2\Build\build\Debug\vc120.pdb'; linking object as if no debug info	C:\devel\nll2\Build\array\blasd.lib(saxpy.obj)

DECLARE_NAMESPACE_NLL

enum CBLAS_ORDER
{
   CblasRowMajor = 101,
   CblasColMajor = 102,
   UnkwownMajor  = 999
};

namespace blas
{
enum CBLAS_TRANSPOSE
{
   CblasNoTrans = 112,
   CblasTrans   = 111
};

using BlasInt        = int;
using BlasReal       = float;
using BlasDoubleReal = double;
}

DECLARE_NAMESPACE_NLL_END
