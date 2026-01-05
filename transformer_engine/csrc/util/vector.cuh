#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp4.h>

#include "common.h"

template <typename T>
struct alignas(2 * sizeof(T)) FPx2 {
  T x;
  T y;
};

template <typename T>
struct FPx4 {
  T x1;
  T x2;
  T x3;
  T x4;
};

template <typename T>
struct Type2x {};

template <>
struct Type2x<float> {
  using type = float2;
};

template <>
struct Type2x<bf16> {
  using type = __nv_bfloat162;
};

template <>
struct Type2x<fp16> {
  using type = __half2;
};

using floatx2 = FPx2<float>;
using bf16x2 = FPx2<bf16>;
using fp16x2 = FPx2<fp16>;
using fp8e4m3x2 = FPx2<fp8e4m3>;
using fp8e5m2x2 = FPx2<fp8e5m2>;

using floatx4 = FPx4<float>;
using bf16x4 = FPx4<bf16>;
using fp16x4 = FPx4<fp16>;
using fp8e4m3x4 = FPx4<fp8e4m3>;
using fp8e5m2x4 = FPx4<fp8e5m2>;

static_assert(sizeof(floatx2) == 8);
static_assert(sizeof(bf16x2) == 4);
static_assert(sizeof(fp16x2) == 4);
static_assert(sizeof(fp8e4m3x2) == 2);
static_assert(sizeof(fp8e5m2x2) == 2);

using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
static_assert(sizeof(fp4e2m1x2) == 1);
static_assert(sizeof(fp4e2m1x4) == 2);

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
  uint4 u;
  uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
  using Type = uint16;
  static_assert(sizeof(Type) == 64);
};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
  enum { BYTES = NUM_ELT * sizeof(Elt_type) };

  using Vec_type = typename BytesToType<BYTES>::Type;
  using type = Elt_type;

  using Alias_type = union {
    Vec_type vec;
    Elt_type elt[NUM_ELT];
  };

  Alias_type data;

  template <typename S>
  inline __device__ void to(Vec<S, NUM_ELT> &other) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      other.data.elt[it] = S(this->data.elt[it]);
    }
  }

  template <typename Op>
  inline __device__ void assign(const Op &op) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = op(it);
    }
  }

  // Pointer is cast to vector type
  inline __device__ void load_from(const void *base_ptr, size_t idx = 0) {
    this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
  }

  // Pointer is cast to vector type
  inline __device__ void store_to(void *base_ptr, size_t idx = 0) const {
    static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
  }

  // Pointer is cast to element type. Loads min(count, NUM_ELT)
  // elements and any remaining elements are set to zero.
  inline __device__ void load_from_elts(const void *base_ptr, size_t idx = 0,
                                        size_t count = NUM_ELT) {
    const Elt_type *elt_ptr = static_cast<const Elt_type *>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
      }
    } else {
      this->load_from(elt_ptr);
    }
  }

  // Pointer is cast to element type. Stores min(count, NUM_ELT)
  // elements.
  inline __device__ void store_to_elts(void *base_ptr, size_t idx = 0,
                                       size_t count = NUM_ELT) const {
    Elt_type *elt_ptr = static_cast<Elt_type *>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        if (it < count) {
          elt_ptr[it] = this->data.elt[it];
        }
      }
    } else {
      this->store_to(elt_ptr);
    }
  }

  inline __device__ void clear() {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = Elt_type(0.f);
    }
  }
};