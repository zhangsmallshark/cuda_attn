#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Standard Library includes
#include <iostream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
template <typename LayoutInputA, typename LayoutInputB>
cudaError_t CutlassSgemm(int M, int N, int K, float alpha, float const *A,
                         int lda, float const *B, int ldb, float beta, float *C,
                         int ldc, float *bias, cudaStream_t stream) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions including the following example for single-precision GEMM.
  // Typical values are used as default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  //   using ColumnMajor = cutlass::layout::ColumnMajor;

  //   using CutlassGemm = cutlass::gemm::device::Gemm<float,        //
  //   Data-type of A matrix
  //                                                   ColumnMajor,  // Layout
  //                                                   of A matrix float, //
  //                                                   Data-type of B matrix
  //                                                   ColumnMajor,  // Layout
  //                                                   of B matrix float, //
  //                                                   Data-type of C matrix
  //                                                   ColumnMajor>; // Layout
  //                                                   of C matrix

  ///////////////////////////////////////////////////////////////////////////////////////////////////

  // The code section below describes datatype for input, output matrices and
  // computation between elements in input matrices.
  using ElementAccumulator = float; // <- data type of accumulator
  using ElementComputeEpilogue =
      ElementAccumulator;      // <- data type of epilogue operations
  using ElementInputA = float; // <- data type of elements in input matrix A
  using ElementInputB = float; // <- data type of elements in input matrix B
  using ElementOutput = float; // <- data type of elements in output matrix D

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C using LayoutInputA = cutlass::layout::RowMajor; using
  // LayoutInputB = cutlass::layout::ColumnMajor; using LayoutOutput =
  // cutlass::layout::RowMajor;

  // using LayoutInputA =
  //     cutlass::layout::RowMajor; // (trans_a)? cutlass::layout::ColumnMajor :
  //                                // cutlass::layout::RowMajor;
  // using LayoutInputB =
  //     cutlass::layout::ColumnMajor; // (trans_b)?
  //     cutlass::layout::ColumnMajor :
  //                                   // cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassSimt;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<256, 128, 16>; // <- threadblock tile M = 256, N
                                              // = 128, K = 16
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<64, 64,
                               16>; // <- warp tile M = 64, N = 64, K = 16
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<1, 1, 1>; // <- MMA Op tile M = 16, N = 8, K = 8

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

  // This code section describes the epilogue part of the kernel
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      1, // 128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number
         // of elements per vectorized
         //  memory access. For a byte, it's 16
         //  elements. This becomes the vector width of
         //  math instructions in the epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                               // combination function

  // Number of pipelines you want to use
  constexpr int NumStages = 4;

  using CutlassGemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm and
  // its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //

  float* c_ptr = bias? bias : C;
  int ldc_b = bias? 0 : ldc;

  typename CutlassGemm::Arguments args(
      {M, N, K}, // Gemm Problem dimensions
      {A, lda},  // Tensor-ref for source matrix A
      {B, ldb},  // Tensor-ref for source matrix B
      {c_ptr, ldc_b}, // Tensor-ref for source matrix C
      {C, ldc},  // Tensor-ref for destination matrix D (may be different memory
                 // than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue


  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = CutlassGemm::get_workspace_size(args);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args, workspace.get(), stream);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}


std::vector<torch::Tensor> linear_cutlass_forward(torch::Tensor input,
                                                  torch::Tensor weights,
                                                  torch::Tensor bias) {
  // input: (batch_size, in_features)
  // weights: (out_features, in_features)
  // bias : (out_features)

  auto in_sizes = input.sizes();
  int batch_size = in_sizes[0];
  int in_features = in_sizes[1];
  int out_features = weights.size(0);

  float alpha = 1.0;
  float beta = 0;

  // Compute leading dimensions for each matrix.
  int lda = in_features;
  int ldb = in_features;
  int ldc = out_features;

  auto one_vec = torch::ones(batch_size);

  cudaStream_t stream0;
  cudaStreamCreate(&stream0);

  // output = input * weights^T (without biases)
  // auto output = torch::zeros_like(input);
  auto output =
      torch::empty({batch_size, out_features},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  CutlassSgemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(
      batch_size, out_features, in_features, alpha, input.data_ptr<float>(),
      lda, weights.data_ptr<float>(), ldb, beta, output.data_ptr<float>(), ldc,
      bias.data_ptr<float>(), stream0);

  // output += biases * one_vec^T
  // float beta = 1.0;
  // result = CutlassSgemm(M, N, K, alpha, bias.data_ptr<float>(), lda,
  // one_vec.data_ptr<float>(), ldb, beta, output.data_ptr<float>(), ldc,
  // stream0);

  return {output};
}

std::vector<torch::Tensor> linear_cutlass_backward(torch::Tensor grad_output,
                                                   torch::Tensor input,
                                                   torch::Tensor weights) {
  auto in_sizes = input.sizes();
  int batch_size = in_sizes[0];
  int in_features = in_sizes[1];
  int out_features = weights.size(0);

  auto grad_weights = torch::zeros_like(weights); // (out_features, in_features)
  auto grad_input = torch::zeros_like(input);     // (batch_size, in_features)

  // int M = 1024;
  // int K = 1024;
  // int N = 1024;

  // int lda = M;
  // int ldb = K;
  // int ldc = M;

  float alpha = 1.0;
  float beta = 0;
  // int batch_size = 1024;
  auto grad_biases = torch::sum(grad_output, 0);

  cudaStream_t stream0;
  cudaStreamCreate(&stream0);

  // db = (dy) * one_vec
  // result = CutlassSgemm(M, N, K, alpha, grad_output.data_ptr<float>(), lda,
  // one_vec.data_ptr<float>(), ldb, beta, grad_biases, ldc, stream0);

  // dw = (dy)^T * x
  CutlassSgemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(
      out_features, in_features, batch_size, alpha,
      grad_output.data_ptr<float>(), out_features, input.data_ptr<float>(),
      in_features, beta, grad_weights.data_ptr<float>(), in_features, nullptr, stream0);

  // dx = dy * w
  CutlassSgemm<cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
      batch_size, in_features, out_features, alpha,
      grad_output.data_ptr<float>(), out_features, weights.data_ptr<float>(),
      in_features, beta, grad_input.data_ptr<float>(), in_features, nullptr, stream0);

  return {grad_input, grad_weights, grad_biases};
}
