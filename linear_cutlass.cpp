#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias);

std::vector<torch::Tensor> linear_cutlass_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> linear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);

  return linear_cutlass_forward(input, weights, bias);
}

std::vector<torch::Tensor> linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return linear_cutlass_backward(
      grad_output,
      input,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "Linear forward (CUDA)");
  m.def("backward", &linear_backward, "Linear backward (CUDA)");
}