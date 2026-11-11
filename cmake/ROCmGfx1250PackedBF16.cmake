# gfx1250 packed-BF16 inline constants (llvm/llvm-project#209861).
#
# LLVM without that fix emits v_pk_{add,mul,fma}_bf16 with an inline-constant
# operand (such as 1.0 or 0.5) but without the op_sel that selects its upper
# 16 bits, so on gfx1250 the constant reads as zero. Any BF16 kernel whose
# vectorized code uses such an encoding computes wrong results; affected ATen
# kernels include grid_sampler_2d bicubic backward, tanh_backward and
# binary_cross_entropy. A compiler version does not show whether a given
# LLVM build has the fix, so compile the smallest affected case for gfx1250
# and look for the fixed encoding (op_sel:[0,1]).
#
# Unless the probe finds it, every HIP translation unit configured after this
# point is compiled without packed BF16 instructions. The feature exists only
# on gfx1250, so other architectures compile unchanged. A failed or
# inconclusive probe also keeps the flag. This covers PyTorch's
# CMake-compiled HIP sources only, not prebuilt libraries or kernels compiled
# at run time. It also drops packed BF16 where it was correct, such as the
# int4 TinyGemm kernels behind _weight_int4pack_mm, which measured 6-17%
# slower on gfx1250 with an affected compiler.
#
# Usage: torch_rocm_gfx1250_bf16_pk_check(<arch list> <compiler>) sets
# TORCH_ROCM_GFX1250_NO_BF16_PK (ON/OFF) in the caller's scope. Conclusive
# results are cached and keyed by the compiler binary's contents, its
# version output, the probe source, command and recognition rule; failed and
# inconclusive results are not cached.

set(_torch_gfx1250_bf16_pk_rule "v1: v_pk_add_bf16 with op_sel:[0,1] is fixed")
set(_torch_gfx1250_bf16_pk_probe
  "target triple = \"amdgcn-amd-amdhsa\"\n"
  "define <2 x bfloat> @probe(<2 x bfloat> %x) {\n"
  "  %r = fadd <2 x bfloat> %x, <bfloat 0xR3F80, bfloat 0xR3F80>\n"
  "  ret <2 x bfloat> %r\n"
  "}\n")

function(torch_rocm_gfx1250_bf16_pk_check arch_list compiler)
  set(TORCH_ROCM_GFX1250_NO_BF16_PK OFF PARENT_SCOPE)
  # Entries reach --offload-arch unchanged, so also split comma-separated
  # lists. Match the base architecture, so target IDs such as gfx1250:sramecc+
  # count. "native" may resolve to gfx1250, so it also runs the probe.
  string(REPLACE "," ";" _archs "${arch_list}")
  set(_has_gfx1250 OFF)
  foreach(_arch IN LISTS _archs)
    string(REGEX REPLACE ":.*$" "" _base "${_arch}")
    if(_base STREQUAL "gfx1250" OR _base STREQUAL "native")
      set(_has_gfx1250 ON)
    endif()
  endforeach()
  if(NOT _has_gfx1250)
    return()
  endif()

  string(CONCAT _probe_src ${_torch_gfx1250_bf16_pk_probe})
  set(_probe "${CMAKE_BINARY_DIR}/CMakeFiles/gfx1250_bf16_pk_probe.ll")
  file(WRITE "${_probe}" "${_probe_src}")
  # -nogpulib: the probe needs no device libraries.
  set(_cmd "${compiler}" -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nogpulib
           -O3 -S -x ir "${_probe}" -o -)

  # Cache key: the resolved compiler's contents and version output, plus the
  # probe source, command and recognition rule.
  get_filename_component(_real "${compiler}" REALPATH)
  set(_key "")
  if(EXISTS "${_real}")
    file(SHA256 "${_real}" _compiler_hash)
    execute_process(COMMAND "${compiler}" --version
                    RESULT_VARIABLE _ver_result OUTPUT_VARIABLE _ver
                    ERROR_QUIET)
    if(_ver_result EQUAL 0)
      string(SHA256 _key "${_compiler_hash}\n${_ver}\n${_probe_src}\n${_cmd}\n${_torch_gfx1250_bf16_pk_rule}")
    endif()
  endif()
  if(_key AND _key STREQUAL "${TORCH_ROCM_GFX1250_BF16_PK_PROBE_KEY}")
    message(STATUS "gfx1250 packed BF16 probe: cached result "
                   "'${TORCH_ROCM_GFX1250_BF16_PK_PROBE_RESULT}' for this compiler")
    if(TORCH_ROCM_GFX1250_BF16_PK_PROBE_RESULT STREQUAL "fixed")
      return()
    endif()
    set(TORCH_ROCM_GFX1250_NO_BF16_PK ON PARENT_SCOPE)
    return()
  endif()

  execute_process(COMMAND ${_cmd}
                  RESULT_VARIABLE _result
                  OUTPUT_VARIABLE _asm
                  ERROR_VARIABLE _err)
  string(REGEX MATCH "v_pk_add_bf16[^\n]*" _insn "${_asm}")
  string(STRIP "${_err}" _err)
  set(_outcome "")
  if(NOT _result EQUAL 0)
    message(STATUS "gfx1250 packed BF16 probe could not run (${_result}): "
                   "${_err}; building HIP sources without packed BF16 "
                   "instructions")
  elseif(_insn STREQUAL "")
    message(STATUS "gfx1250 packed BF16 probe inconclusive: no v_pk_add_bf16 "
                   "in its output; building HIP sources without packed BF16 "
                   "instructions")
  elseif(_insn MATCHES "op_sel:\\[0,1\\]")
    set(_outcome "fixed")
    message(STATUS "gfx1250 packed BF16 probe: inline constants are encoded "
                   "correctly; HIP sources keep packed BF16 instructions")
  else()
    set(_outcome "broken")
    message(STATUS "gfx1250 packed BF16 probe: the compiler lacks "
                   "llvm/llvm-project#209861 (\"${_insn}\"); building HIP "
                   "sources without packed BF16 instructions")
  endif()
  if(_key AND _outcome)
    set(TORCH_ROCM_GFX1250_BF16_PK_PROBE_KEY "${_key}" CACHE INTERNAL
        "Compiler and probe identity of the cached gfx1250 packed BF16 result")
    set(TORCH_ROCM_GFX1250_BF16_PK_PROBE_RESULT "${_outcome}" CACHE INTERNAL
        "Cached gfx1250 packed BF16 probe result (fixed or broken)")
  else()
    unset(TORCH_ROCM_GFX1250_BF16_PK_PROBE_KEY CACHE)
    unset(TORCH_ROCM_GFX1250_BF16_PK_PROBE_RESULT CACHE)
  endif()
  if(NOT _outcome STREQUAL "fixed")
    set(TORCH_ROCM_GFX1250_NO_BF16_PK ON PARENT_SCOPE)
  endif()
endfunction()
