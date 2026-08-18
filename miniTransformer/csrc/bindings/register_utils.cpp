#include "bindings/register_all.h"
#include "ops.h"

// [UTIL]
void register_utils(py::module& m) {
    m.def("swizzle_scaling_factors", &swizzle_scaling_factors, "Swizzle scaling factors (CUDA)");
}
