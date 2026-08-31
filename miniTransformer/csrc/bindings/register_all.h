#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_utils(py::module& m);
void register_fp4_quantize(py::module& m);
void register_fp4_gemm(py::module& m);
void register_fp8_quantize(py::module& m);
void register_fp8_gemm(py::module& m);
void register_fp4_quarter(py::module& m);
