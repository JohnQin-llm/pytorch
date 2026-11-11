#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/util/strides.h>

#include <algorithm>
#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

inline bool check_valid_strides_and_return_transposed(const Tensor& mat) {
  IntArrayRef tensor_strides = mat.strides();
  IntArrayRef tensor_sizes = mat.sizes();
  int end_dim = mat.dim() - 1;
  int alignment = 16 / mat.element_size();
  bool is_cpu = mat.device().is_cpu();
  TORCH_CHECK(is_cpu || uint64_t(mat.data_ptr()) % 16 == 0, "expected data_ptr to be aligned to 16 bytes");
  // 3D inputs: per-batch pointer is base + i*stride(0)*element_size. cuBLAS and cutlass
  // grouped GEMM kernels use TMA which requires 16-byte-aligned pointers, so
  // the per-batch step must preserve the alignment of the base pointer.
  if (!is_cpu && mat.dim() == 3) {
    TORCH_CHECK(tensor_strides[0] % alignment == 0,
        "batch stride must yield 16-byte-aligned per-batch pointers, got stride(0)=",
        tensor_strides[0], " with element size ", mat.element_size(),
        " (sizes=", tensor_sizes, ", strides=", tensor_strides, ")");
  }
  if ((tensor_strides[end_dim - 1] == 1) && (tensor_strides[end_dim] >= std::max<int64_t>(1, tensor_sizes[end_dim - 1]))) {
    TORCH_CHECK(tensor_strides[end_dim] % alignment == 0, "strides should be multiple of 16 bytes");
    return true;
  } else if ((tensor_strides[end_dim] == 1) && (tensor_strides[end_dim - 1] >= std::max<int64_t>(1, tensor_sizes[end_dim]))) {
    TORCH_CHECK(tensor_strides[end_dim - 1] % alignment == 0, "strides should be multiple of 16 bytes");
    return false;
  } else {
    TORCH_CHECK(false, "Invalid strides/sizes, got ", mat.strides(), " for strides and ", mat.sizes(), " for sizes");
  }
}

// Compute the size and stride of a grouped-gemm output. Strides match the
// layout `create_grouped_gemm_output_tensor` allocates: on CUDA the last dim
// is padded to a 16-byte-aligned stride for TMA transfers; on ROCm the
// returned stride vector is empty, denoting default contiguous strides (the
// allocator uses at::empty/at::zeros there). Shared by the eager allocator and
// the structured TORCH_META_FUNC so the shape/stride math stays single-sourced.
inline std::pair<c10::SmallVector<int64_t, 3>, std::vector<int64_t>>
compute_grouped_gemm_output_size_stride(const Tensor& mat_a,
const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
c10::ScalarType out_dtype
) {
  c10::SmallVector<int64_t, 3> out_size;
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (a_is_2d) {
    if (b_is_2d) {
      out_size = {offs->size(0), mat_a.size(0), mat_b.size(1)};
    } else {
      TORCH_CHECK(offs->size(0) == mat_b.size(0), "matrix batch sizes have to match");
      out_size = {mat_a.size(0), mat_b.size(-1)};
    }
  } else {
    if (b_is_2d) {
      // this case is not actually encountered for MoE gemms
      TORCH_CHECK(offs->size(0) == mat_a.size(0), "matrix batch sizes have to match");
      out_size = {mat_a.size(1), mat_b.size(1)};
    } else { // regular bmm
      TORCH_CHECK(mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
      out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
    }
  }

  std::vector<int64_t> out_stride;
  #ifndef USE_ROCM
  // For TMA transfers, strides of output tensor have to be either 1, or aligned
  // to 16 bytes. Pad the last dim up to that alignment and take the contiguous
  // strides of the padded shape (so the second-to-last stride picks up the pad).
  const auto last_dim = out_size.size() - 1;
  const auto alignment = 16 / c10::elementSize(out_dtype);
  auto padded_size = out_size;
  padded_size[last_dim] = (out_size[last_dim] + alignment - 1) / alignment * alignment;
  const auto strides = c10::contiguous_strides(padded_size);
  out_stride.assign(strides.begin(), strides.end());
  #endif
  return {out_size, out_stride};
}

inline at::Tensor create_grouped_gemm_output_tensor(const Tensor& mat_a,
const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
c10::ScalarType out_dtype
) {
  auto [out_size, out_stride] =
      compute_grouped_gemm_output_size_stride(mat_a, mat_b, offs, out_dtype);

  #ifndef USE_ROCM
  return at::empty_strided(out_size, out_stride, mat_a.options().dtype(out_dtype));
  #else
  // For ROCm 2D-2D case (output is 3D), zero-initialize to handle K=0 or small K
  // groups correctly. When K=0, the mathematically correct result is zeros,
  // but CK kernel may not write to the output region.
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    return at::zeros(out_size, mat_a.options().dtype(out_dtype));
  }
  return at::empty(out_size, mat_a.options().dtype(out_dtype));
  #endif
}

inline void _grouped_mm_validate_inputs(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  TORCH_CHECK((mat_a.dtype() == at::kBFloat16) || (mat_a.dtype() == at::kFloat) || (mat_a.dtype() == at::kHalf), "Expected mat_a to be Float32, BFloat16 or Float16 matrix, got ", mat_a.scalar_type());
  TORCH_CHECK((mat_b.dtype() == at::kBFloat16) || (mat_b.dtype() == at::kFloat) || (mat_b.dtype() == at::kHalf), "Expected mat_b to be Float32, BFloat16 or Float16 matrix, got ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.dtype() == mat_b.dtype(), "Expected mat_a and mat_b to have the same dtype, got ", mat_a.scalar_type(), " and ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }

  // check that the strides are valid, the fn will throw an error if not
  check_valid_strides_and_return_transposed(mat_a);
  check_valid_strides_and_return_transposed(mat_b);
  TORCH_CHECK(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
    TORCH_CHECK(offs->stride(0) == 1, "Offsets must have stride (1,)");
  }
  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");
}

inline c10::ScalarType _resolve_grouped_mm_out_dtype(const Tensor& mat_a, [[maybe_unused]] const Tensor& mat_b,
std::optional<c10::ScalarType> out_dtype) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  // TODO(future PR): enable float32 output dtype for bfloat16 and float16 inputs
  TORCH_CHECK(out_dtype_ == mat_a.dtype(), "Grouped gemm output dtype must match `mat_a` dtype");
  return out_dtype_;
}


#ifdef USE_ROCM
// Runs a 2d x 3d grouped mm (MoE: rows sorted by expert x per-expert weights) as one batched GEMM.
// The fallback loop launches one GEMM per group, and on ROCm each of them costs several microseconds
// however small it is, so with dozens to hundreds of small groups the loop reaches a few percent of
// peak. When the groups are similar in size, gathering them into a buffer padded to the largest group
// and running a single bmm is much faster. Returns false, having done nothing, when the padding is
// expected to cost more than it saves.
inline bool _grouped_mm_2d_3d_as_padded_bmm(const Tensor& mat_a, const Tensor& mat_b,
const Tensor& offs,
const Tensor& offs_cpu,
Tensor& out) {
  const int64_t num_groups = offs_cpu.size(0);
  // The output is written through row-packed views (ldc == N, packed batch stride). ROCm allocates it
  // contiguous; anything else stays on the loop.
  if (!mat_a.is_cuda() || num_groups < 16 || out.scalar_type() != mat_a.scalar_type() || !out.is_contiguous()) {
    return false;
  }
  const int32_t* group_ends = offs_cpu.const_data_ptr<int32_t>();
  int64_t total_rows = 0;
  int64_t max_rows = 0;
  for (int64_t group_idx = 0; group_idx < num_groups; group_idx++) {
    const int64_t rows = static_cast<int64_t>(group_ends[group_idx]) - total_rows;
    if (rows < 0) {
      return false;
    }
    max_rows = std::max(max_rows, rows);
    total_rows = group_ends[group_idx];
  }
  // Measured on MI355X over MoE shapes (8 to 256 groups, K and N from 768 to 28672, balanced and skewed
  // routing): with at least 16 groups, at most 512 rows per group on average and at most 3x padding, one
  // bmm plus the gather and scatter is 1.1x to 8.8x faster than the loop. With fewer or longer groups, or
  // more padding, the loop can be faster.
  if (total_rows == 0 || total_rows > mat_a.size(0) || total_rows > 512 * num_groups ||
      num_groups * max_rows > 3 * total_rows) {
    return false;
  }
  const int64_t K = mat_a.size(1);
  const int64_t N = mat_b.size(2);
  const auto a_used = mat_a.narrow(0, 0, total_rows);
  auto out_used = out.narrow(0, 0, total_rows);
  if (num_groups * max_rows == total_rows) {
    // All groups have the same size, so no padding is needed.
    auto out_batched = out_used.view({num_groups, max_rows, N});
    at::bmm_out(out_batched, a_used.reshape({num_groups, max_rows, K}), mat_b);
    return true;
  }
  // Row r of group g goes to row g * max_rows + (r - start of group g) of the padded buffer.
  const auto offs_long = offs.to(at::kLong);
  const auto group_rows = offs_long.diff(1, 0, offs_long.new_zeros({1}));
  const auto row_group = at::repeat_interleave(group_rows, /*output_size=*/std::optional<int64_t>(total_rows));
  const auto group_shift =
      at::arange(num_groups, offs_long.options()).mul_(max_rows).sub_(offs_long).add_(group_rows);
  const auto padded_row = at::arange(total_rows, offs_long.options()).add_(group_shift.index_select(0, row_group));
  // The padding rows stay uninitialized: they only produce output rows that are never read.
  auto a_padded = at::empty({num_groups * max_rows, K}, mat_a.options());
  a_padded.index_copy_(0, padded_row, a_used);
  const auto out_padded = at::bmm(a_padded.view({num_groups, max_rows, K}), mat_b);
  at::index_select_out(out_used, out_padded.view({num_groups * max_rows, N}), 0, padded_row);
  return true;
}
#endif

inline void _grouped_mm_fallback(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype,
Tensor out) {
  LOG(INFO) << "fallback path for `torch._grouped_mm`, performance may not be optimal";
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (a_is_2d && !b_is_2d) {
    // 2d x 3d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
#ifdef USE_ROCM
    if (_grouped_mm_2d_3d_as_padded_bmm(mat_a, mat_b, offs.value(), offs_cpu, out)) {
      return;
    }
#endif
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_a_slice = mat_a.slice(0, group_start_idx, group_end_idx);
      auto out_slice = out.slice(0, group_start_idx, group_end_idx);
      at::mm_out(out_slice, mat_a_slice, mat_b[group_idx]);
      group_start_idx = group_end_idx;
    }

  } else if (!a_is_2d && b_is_2d) {
    // 3d x 2d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_b_slice = mat_b.slice(1, group_start_idx, group_end_idx);
      auto out_slice = out.slice(1, group_start_idx, group_end_idx);
      at::mm_out(out_slice, mat_a[group_idx], mat_b_slice);
      group_start_idx = group_end_idx;
    }

  } else if (a_is_2d && b_is_2d) {
    // 2d x 2d with offsets
    int group_start_idx = 0;
    auto offs_cpu = offs.value().cpu();
    for (int group_idx = 0; group_idx < offs_cpu.size(0); group_idx++) {
      int group_end_idx = offs_cpu[group_idx].item<int>();
      auto mat_a_slice = mat_a.slice(1, group_start_idx, group_end_idx);
      auto mat_b_slice = mat_b.slice(0, group_start_idx, group_end_idx);
      auto out_slice = out[group_idx];
      at::mm_out(out_slice, mat_a_slice, mat_b_slice);
      group_start_idx = group_end_idx;
    }

  } else {
    // 3d x 3d without offsets - regular bmm
    at::bmm_out(out, mat_a, mat_b);
  }
}


} // namespace at::native
