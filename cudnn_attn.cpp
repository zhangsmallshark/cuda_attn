#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>

void initData(std::vector<float>& container)
{
    int count = 0;
    for (std::vector<int>::size_type i = 0; i < container.size(); i++) {
        container[i] = count;
        count++;
    }
}

void initDataOne(std::vector<float>& container)
{
    int count = 1;
    for (std::vector<int>::size_type i = 0; i < container.size(); i++) {
        container[i] = count;
    }
}

void checkCUDNN(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
        std::cout << "[ERROR] CUDNN " << cudnnGetErrorString(status) << std::endl;
}

void checkCUDNN(cudnnStatus_t status, std::string pos)
{
    if (status != CUDNN_STATUS_SUCCESS)
        std::cout << "[ERROR] CUDNN " << cudnnGetErrorString(status) << " " << pos << std::endl;
}

void checkCUDA(cudaError_t error)
{
    if (error != CUDA_SUCCESS)
        std::cout << "[ERROR] CUDA " << error << std::endl;
}


// int main() {

//     int num_heads = 1;
//     int hidden_dim_per_head = 64;
//     int hidden_dim = num_heads * hidden_dim_per_head;
//     int max_seq_len = 1024;
//     int batch_size = 16;

//     // std::vector<float> q(hidden_dim * max_seq_len);
//     // std::vector<float> k(hidden_dim * max_seq_len);
//     // std::vector<float> k(hidden_dim * max_seq_len);
//     // initData(q);
//     // initData(k);
//     // initData(v);

//     std::vector<float> in_seq(batch_size * hidden_dim * max_seq_len);
//     initData(in_seq);

//     cudaSetDevice(0); 
//     cudaStream_t stream;
//     cudnnHandle_t cudnn_handle;
//     checkCUDA(cudaStreamCreate(&stream));
//     checkCUDNN(cudnnCreate(&cudnn_handle));
//     checkCUDNN(cudnnSetStream(cudnn_handle, stream));

//     float* d_in_seq = nullptr;
//     checkCUDA(cudaMalloc(&d_in_seq, in_seq.size() * sizeof(float)));
//     checkCUDA(cudaMemcpy(d_in_seq, in_seq.data(), in_seq.size() * sizeof(float), cudaMemcpyHostToDevice));
//     float* q = d_in_seq;
//     float* k = d_in_seq;
//     float* v = d_in_seq;

//     float* d_out_seq = nullptr;
//     checkCUDA(cudaMalloc(&d_out_seq, hidden_dim * max_seq_len * sizeof(float)));

//     cudnnAttnDescriptor_t attn_desc;
//     cudnnCreateAttnDescriptor(&attn_desc);
//     // cudnnDestroyAttnDescriptor()

//     double softmax_scaler = 1.0 / (sqrt(hidden_dim_per_head));

//     cudnnDropoutDescriptor_t dropout_desc;
//     cudnnCreateDropoutDescriptor(&dropout_desc);
//     float dropout = 0.1;
//     unsigned long long seed = 668;
//     size_t state_size;
//     cudnnDropoutGetStatesSize(
//         /*cudnnHandle_t       handle,*/ cudnn_handle, 
//         /*size_t             *sizeInBytes*/ &state_size);
//     float* d_states = nullptr;
//     checkCUDA(cudaMalloc(&d_states, state_size));
//     checkCUDNN(cudnnSetDropoutDescriptor(
//         /*cudnnDropoutDescriptor_t    dropoutDesc,*/ dropout_desc,
//         /*cudnnHandle_t               handle,*/ cudnn_handle,
//         /*float                       dropout,*/ dropout,
//         /*void                       *states,*/ d_states,
//         /*size_t                      stateSizeInBytes,*/ state_size,
//         /*unsigned long long          seed*/ seed), "cudnnSetDropoutDescriptor");

//     checkCUDNN(cudnnSetAttnDescriptor(
//         /*cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
//         /*unsigned attnMode,*/ CUDNN_ATTN_DISABLE_PROJ_BIASES,
//         /*int nHeads,*/ num_heads,
//         /*double smScaler,*/ softmax_scaler,
//         /*cudnnDataType_t dataType,*/ CUDNN_DATA_FLOAT,
//         /*cudnnDataType_t computePrec,*/ CUDNN_DATA_FLOAT,
//         /*cudnnMathType_t mathType,*/ CUDNN_DEFAULT_MATH,
//         /*cudnnDropoutDescriptor_t attnDropoutDesc,*/ dropout_desc,
//         /*cudnnDropoutDescriptor_t postDropoutDesc,*/ dropout_desc,
//         /*int qSize,*/ hidden_dim_per_head,
//         /*int kSize,*/ hidden_dim_per_head,
//         /*int vSize,*/ hidden_dim_per_head,
//         /*int qProjSize,*/ hidden_dim_per_head,
//         /*int kProjSize,*/ hidden_dim_per_head,
//         /*int vProjSize,*/ hidden_dim_per_head,
//         /*int oProjSize,*/ hidden_dim_per_head,
//         /*int qoMaxSeqLength,*/ max_seq_len,
//         /*int kvMaxSeqLength,*/ max_seq_len,
//         /*int maxBatchSize,*/batch_size,
//         /*int maxBeamSize  */1), "cudnnSetAttnDescriptor");

//     /*------------------------------------------------------*/
//     cudnnSeqDataDescriptor_t seq_data_desc;
//     cudnnCreateSeqDataDescriptor(&seq_data_desc);
//     // cudnnDestroySeqDataDescriptor();

//     int dim_a[CUDNN_SEQDATA_DIM_COUNT];
//     dim_a[CUDNN_SEQDATA_TIME_DIM] = max_seq_len;
//     dim_a[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
//     dim_a[CUDNN_SEQDATA_BEAM_DIM] = 1;
//     dim_a[CUDNN_SEQDATA_VECT_DIM] = hidden_dim_per_head;

//     cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
//     axes[3] = CUDNN_SEQDATA_VECT_DIM;
//     axes[2] = CUDNN_SEQDATA_BEAM_DIM;
//     axes[1] = CUDNN_SEQDATA_BATCH_DIM;
//     axes[0] = CUDNN_SEQDATA_TIME_DIM;

//     size_t seq_len_arr_size = batch_size;
//     std::vector<int> seq_len_vec(batch_size, max_seq_len);
//     int* seq_len_arr = &seq_len_vec[0];
//     // int seq_len_arr[batch_size];

//     checkCUDNN(cudnnSetSeqDataDescriptor(
//         /*cudnnSeqDataDescriptor_t seqDataDesc,*/ seq_data_desc,
//         /*cudnnDataType_t dataType,*/ CUDNN_DATA_FLOAT,
//         /*int nbDims,*/ 4,
//         /*const int dimA[],*/ dim_a,
//         /*const cudnnSeqDataAxis_t axes[],*/ axes,
//         /*size_t seqLengthArraySize,*/  seq_len_arr_size,
//         /*const int seqLengthArray[],*/ seq_len_arr,
//         /*void *paddingFill*/ NULL));

//     /*------------------------------------------------------*/
//     size_t weight_size;
//     size_t workspace_size;
//     size_t reserve_space_size;
//     checkCUDNN(cudnnGetMultiHeadAttnBuffers(
//         /*cudnnHandle_t handle,*/ cudnn_handle, 
//         /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
//         /*size_t *weightSizeInBytes,*/    &weight_size,
//         /*size_t *workSpaceSizeInBytes,*/ &workspace_size,
//         /*size_t *reserveSpaceSizeInBytes*/ &reserve_space_size));

//     std::vector<float> weight(weight_size / 4);
//     initDataOne(weight);
//     float* d_weight; // device weight
//     checkCUDA(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
//     checkCUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

//     float* d_workspace = nullptr;
//     checkCUDA(cudaMalloc(&d_workspace, workspace_size * 1));
//     float* d_reserve_space = nullptr;
//     checkCUDA(cudaMalloc(&d_reserve_space, reserve_space_size * 1));

//     /*------------------------------------------------------*/
//     std::vector<int> lo_win_idx_vec(max_seq_len, 0);
//     std::vector<int> hi_win_idx_vec(max_seq_len, max_seq_len);
//     int* lo_win_idx = &lo_win_idx_vec[0];
//     int* hi_win_idx = &hi_win_idx_vec[0];

//     std::vector<int> seq_len_q_o{max_seq_len, max_seq_len};
//     int* d_seq_len_q_o = nullptr;
//     checkCUDA(cudaMalloc(&d_seq_len_q_o, seq_len_q_o.size() * sizeof(int)));
//     checkCUDA(cudaMemcpy(d_seq_len_q_o, seq_len_q_o.data(), seq_len_q_o.size() * sizeof(int), cudaMemcpyHostToDevice));

//     std::vector<int> seq_len_k_v{max_seq_len, max_seq_len};
//     int* d_seq_len_k_v = nullptr;
//     checkCUDA(cudaMalloc(&d_seq_len_k_v, seq_len_k_v.size() * sizeof(int)));
//     checkCUDA(cudaMemcpy(d_seq_len_k_v, seq_len_k_v.data(), seq_len_k_v.size() * sizeof(int), cudaMemcpyHostToDevice));

//     checkCUDNN(cudnnMultiHeadAttnForward(
//         /*cudnnHandle_t handle,*/ cudnn_handle,
//         /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
//         /*int currIdx,*/ -1,
//         /*const int loWinIdx[],*/ lo_win_idx,
//         /*const int hiWinIdx[],*/ hi_win_idx,
//         /*const int devSeqLengthsQO[],*/ d_seq_len_q_o,
//         /*const int devSeqLengthsKV[],*/ d_seq_len_k_v,
//         /*const cudnnSeqDataDescriptor_t qDesc,*/ seq_data_desc,
//         /*const void *queries,*/ q,
//         /*const void *residuals,*/ q,
//         /*const cudnnSeqDataDescriptor_t kDesc,*/ seq_data_desc,
//         /*const void *keys,*/ k,
//         /*const cudnnSeqDataDescriptor_t vDesc,*/ seq_data_desc,
//         /*const void *values,*/ v,
//         /*const cudnnSeqDataDescriptor_t oDesc,*/ seq_data_desc,
//         /*void *out,*/ d_out_seq,
//         /*size_t weightSizeInBytes,*/ weight_size,
//         /*const void *weights,*/ d_weight,
//         /*size_t workSpaceSizeInBytes,*/ workspace_size,
//         /*void *workSpace,*/ d_workspace,
//         /*size_t reserveSpaceSizeInBytes,*/ reserve_space_size,
//         /*void *reserveSpace*/  d_reserve_space), "cudnnMultiHeadAttnForward");

//     std::cout << "finish !!!" << std::endl;
// }

int func1() {

    int num_heads = 1;
    int hidden_dim_per_head = 64;
    int hidden_dim = num_heads * hidden_dim_per_head;
    int max_seq_len = 1024;
    int batch_size = 16;

    // std::vector<float> q(hidden_dim * max_seq_len);
    // std::vector<float> k(hidden_dim * max_seq_len);
    // std::vector<float> k(hidden_dim * max_seq_len);
    // initData(q);
    // initData(k);
    // initData(v);

    std::vector<float> in_seq(batch_size * hidden_dim * max_seq_len);
    initData(in_seq);

    cudaSetDevice(0); 
    cudaStream_t stream;
    cudnnHandle_t cudnn_handle;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));

    cudaStream_t stream1;
    cudnnHandle_t cudnn_handle1;
    checkCUDA(cudaStreamCreate(&stream1));
    checkCUDNN(cudnnCreate(&cudnn_handle1));
    checkCUDNN(cudnnSetStream(cudnn_handle1, stream1));

    float* d_in_seq = nullptr;
    checkCUDA(cudaMalloc(&d_in_seq, in_seq.size() * sizeof(float)));
    checkCUDA(cudaMemcpy(d_in_seq, in_seq.data(), in_seq.size() * sizeof(float), cudaMemcpyHostToDevice));
    float* q = d_in_seq;
    float* k = d_in_seq;
    float* v = d_in_seq;

    float* d_in_seq1 = nullptr;
    checkCUDA(cudaMalloc(&d_in_seq1, in_seq.size() * sizeof(float)));
    checkCUDA(cudaMemcpy(d_in_seq1, in_seq.data(), in_seq.size() * sizeof(float), cudaMemcpyHostToDevice));
    float* q1 = d_in_seq1;
    float* k1 = d_in_seq1;
    float* v1 = d_in_seq1;

    float* d_out_seq = nullptr;
    checkCUDA(cudaMalloc(&d_out_seq, hidden_dim * max_seq_len * sizeof(float)));

    float* d_out_seq1 = nullptr;
    checkCUDA(cudaMalloc(&d_out_seq1, hidden_dim * max_seq_len * sizeof(float)));

    cudnnAttnDescriptor_t attn_desc;
    cudnnCreateAttnDescriptor(&attn_desc);

    cudnnAttnDescriptor_t attn_desc1;
    cudnnCreateAttnDescriptor(&attn_desc1);
    // cudnnDestroyAttnDescriptor()

    double softmax_scaler = 1.0 / (sqrt(hidden_dim_per_head));

    cudnnDropoutDescriptor_t dropout_desc;
    cudnnCreateDropoutDescriptor(&dropout_desc);
    float dropout = 0.1;
    unsigned long long seed = 668;
    size_t state_size;
    cudnnDropoutGetStatesSize(
        /*cudnnHandle_t       handle,*/ cudnn_handle, 
        /*size_t             *sizeInBytes*/ &state_size);
    float* d_states = nullptr;
    checkCUDA(cudaMalloc(&d_states, state_size));
    checkCUDNN(cudnnSetDropoutDescriptor(
        /*cudnnDropoutDescriptor_t    dropoutDesc,*/ dropout_desc,
        /*cudnnHandle_t               handle,*/ cudnn_handle,
        /*float                       dropout,*/ dropout,
        /*void                       *states,*/ d_states,
        /*size_t                      stateSizeInBytes,*/ state_size,
        /*unsigned long long          seed*/ seed), "cudnnSetDropoutDescriptor");

    // cudnn_handle1
    cudnnDropoutDescriptor_t dropout_desc1;
    cudnnCreateDropoutDescriptor(&dropout_desc1);
    size_t state_size1;
    cudnnDropoutGetStatesSize(
        /*cudnnHandle_t       handle,*/ cudnn_handle1, 
        /*size_t             *sizeInBytes*/ &state_size1);
    float* d_states1 = nullptr;
    checkCUDA(cudaMalloc(&d_states1, state_size1));
    checkCUDNN(cudnnSetDropoutDescriptor(
        /*cudnnDropoutDescriptor_t    dropoutDesc,*/ dropout_desc1,
        /*cudnnHandle_t               handle,*/ cudnn_handle1,
        /*float                       dropout,*/ dropout,
        /*void                       *states,*/ d_states1,
        /*size_t                      stateSizeInBytes,*/ state_size1,
        /*unsigned long long          seed*/ seed), "cudnnSetDropoutDescriptor");


    checkCUDNN(cudnnSetAttnDescriptor(
        /*cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
        /*unsigned attnMode,*/ CUDNN_ATTN_DISABLE_PROJ_BIASES,
        /*int nHeads,*/ num_heads,
        /*double smScaler,*/ softmax_scaler,
        /*cudnnDataType_t dataType,*/ CUDNN_DATA_FLOAT,
        /*cudnnDataType_t computePrec,*/ CUDNN_DATA_FLOAT,
        /*cudnnMathType_t mathType,*/ CUDNN_DEFAULT_MATH,
        /*cudnnDropoutDescriptor_t attnDropoutDesc,*/ dropout_desc,
        /*cudnnDropoutDescriptor_t postDropoutDesc,*/ dropout_desc,
        /*int qSize,*/ hidden_dim_per_head,
        /*int kSize,*/ hidden_dim_per_head,
        /*int vSize,*/ hidden_dim_per_head,
        /*int qProjSize,*/ hidden_dim_per_head,
        /*int kProjSize,*/ hidden_dim_per_head,
        /*int vProjSize,*/ hidden_dim_per_head,
        /*int oProjSize,*/ hidden_dim_per_head,
        /*int qoMaxSeqLength,*/ max_seq_len,
        /*int kvMaxSeqLength,*/ max_seq_len,
        /*int maxBatchSize,*/batch_size,
        /*int maxBeamSize  */1), "cudnnSetAttnDescriptor");

    checkCUDNN(cudnnSetAttnDescriptor(
        /*cudnnAttnDescriptor_t attnDesc,*/ attn_desc1,
        /*unsigned attnMode,*/ CUDNN_ATTN_DISABLE_PROJ_BIASES,
        /*int nHeads,*/ num_heads,
        /*double smScaler,*/ softmax_scaler,
        /*cudnnDataType_t dataType,*/ CUDNN_DATA_FLOAT,
        /*cudnnDataType_t computePrec,*/ CUDNN_DATA_FLOAT,
        /*cudnnMathType_t mathType,*/ CUDNN_DEFAULT_MATH,
        /*cudnnDropoutDescriptor_t attnDropoutDesc,*/ dropout_desc1,
        /*cudnnDropoutDescriptor_t postDropoutDesc,*/ dropout_desc1,
        /*int qSize,*/ hidden_dim_per_head,
        /*int kSize,*/ hidden_dim_per_head,
        /*int vSize,*/ hidden_dim_per_head,
        /*int qProjSize,*/ hidden_dim_per_head,
        /*int kProjSize,*/ hidden_dim_per_head,
        /*int vProjSize,*/ hidden_dim_per_head,
        /*int oProjSize,*/ hidden_dim_per_head,
        /*int qoMaxSeqLength,*/ max_seq_len,
        /*int kvMaxSeqLength,*/ max_seq_len,
        /*int maxBatchSize,*/batch_size,
        /*int maxBeamSize  */1), "cudnnSetAttnDescriptor1");

    /*------------------------------------------------------*/
    cudnnSeqDataDescriptor_t seq_data_desc;
    cudnnCreateSeqDataDescriptor(&seq_data_desc);
    // cudnnDestroySeqDataDescriptor();

    int dim_a[CUDNN_SEQDATA_DIM_COUNT];
    dim_a[CUDNN_SEQDATA_TIME_DIM] = max_seq_len;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
    dim_a[CUDNN_SEQDATA_BEAM_DIM] = 1;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = hidden_dim_per_head;

    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    axes[3] = CUDNN_SEQDATA_VECT_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[1] = CUDNN_SEQDATA_BATCH_DIM;
    axes[0] = CUDNN_SEQDATA_TIME_DIM;

    size_t seq_len_arr_size = batch_size;
    std::vector<int> seq_len_vec(batch_size, max_seq_len);
    int* seq_len_arr = &seq_len_vec[0];
    // int seq_len_arr[batch_size];

    checkCUDNN(cudnnSetSeqDataDescriptor(
        /*cudnnSeqDataDescriptor_t seqDataDesc,*/ seq_data_desc,
        /*cudnnDataType_t dataType,*/ CUDNN_DATA_FLOAT,
        /*int nbDims,*/ 4,
        /*const int dimA[],*/ dim_a,
        /*const cudnnSeqDataAxis_t axes[],*/ axes,
        /*size_t seqLengthArraySize,*/  seq_len_arr_size,
        /*const int seqLengthArray[],*/ seq_len_arr,
        /*void *paddingFill*/ NULL));

    /*------------------------------------------------------*/
    size_t weight_size;
    size_t workspace_size;
    size_t reserve_space_size;
    checkCUDNN(cudnnGetMultiHeadAttnBuffers(
        /*cudnnHandle_t handle,*/ cudnn_handle, 
        /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
        /*size_t *weightSizeInBytes,*/    &weight_size,
        /*size_t *workSpaceSizeInBytes,*/ &workspace_size,
        /*size_t *reserveSpaceSizeInBytes*/ &reserve_space_size));

    std::vector<float> weight(weight_size / 4);
    initDataOne(weight);
    float* d_weight; // device weight
    checkCUDA(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
    checkCUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_workspace = nullptr;
    checkCUDA(cudaMalloc(&d_workspace, workspace_size));
    float* d_reserve_space = nullptr;
    checkCUDA(cudaMalloc(&d_reserve_space, reserve_space_size));

    size_t weight_size1;
    size_t workspace_size1;
    size_t reserve_space_size1;
    checkCUDNN(cudnnGetMultiHeadAttnBuffers(
        /*cudnnHandle_t handle,*/ cudnn_handle1, 
        /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc1,
        /*size_t *weightSizeInBytes,*/    &weight_size1,
        /*size_t *workSpaceSizeInBytes,*/ &workspace_size1,
        /*size_t *reserveSpaceSizeInBytes*/ &reserve_space_size1));

    std::vector<float> weight1(weight_size1 / 4);
    initDataOne(weight1);
    float* d_weight1; // device weight
    checkCUDA(cudaMalloc(&d_weight1, weight1.size() * sizeof(float)));
    checkCUDA(cudaMemcpy(d_weight1, weight1.data(), weight1.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_workspace1 = nullptr;
    checkCUDA(cudaMalloc(&d_workspace1, workspace_size1));
    float* d_reserve_space1 = nullptr;
    checkCUDA(cudaMalloc(&d_reserve_space1, reserve_space_size1));

    /*------------------------------------------------------*/
    std::vector<int> lo_win_idx_vec(max_seq_len, 0);
    std::vector<int> hi_win_idx_vec(max_seq_len, max_seq_len);
    int* lo_win_idx = &lo_win_idx_vec[0];
    int* hi_win_idx = &hi_win_idx_vec[0];

    std::vector<int> seq_len_q_o{max_seq_len, max_seq_len};
    int* d_seq_len_q_o = nullptr;
    checkCUDA(cudaMalloc(&d_seq_len_q_o, seq_len_q_o.size() * sizeof(int)));
    checkCUDA(cudaMemcpy(d_seq_len_q_o, seq_len_q_o.data(), seq_len_q_o.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> seq_len_k_v{max_seq_len, max_seq_len};
    int* d_seq_len_k_v = nullptr;
    checkCUDA(cudaMalloc(&d_seq_len_k_v, seq_len_k_v.size() * sizeof(int)));
    checkCUDA(cudaMemcpy(d_seq_len_k_v, seq_len_k_v.data(), seq_len_k_v.size() * sizeof(int), cudaMemcpyHostToDevice));

    checkCUDNN(cudnnMultiHeadAttnForward(
        /*cudnnHandle_t handle,*/ cudnn_handle,
        /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc,
        /*int currIdx,*/ -1,
        /*const int loWinIdx[],*/ lo_win_idx,
        /*const int hiWinIdx[],*/ hi_win_idx,
        /*const int devSeqLengthsQO[],*/ d_seq_len_q_o,
        /*const int devSeqLengthsKV[],*/ d_seq_len_k_v,
        /*const cudnnSeqDataDescriptor_t qDesc,*/ seq_data_desc,
        /*const void *queries,*/ q,
        /*const void *residuals,*/ q,
        /*const cudnnSeqDataDescriptor_t kDesc,*/ seq_data_desc,
        /*const void *keys,*/ k,
        /*const cudnnSeqDataDescriptor_t vDesc,*/ seq_data_desc,
        /*const void *values,*/ v,
        /*const cudnnSeqDataDescriptor_t oDesc,*/ seq_data_desc,
        /*void *out,*/ d_out_seq,
        /*size_t weightSizeInBytes,*/ weight_size,
        /*const void *weights,*/ d_weight,
        /*size_t workSpaceSizeInBytes,*/ workspace_size,
        /*void *workSpace,*/ d_workspace,
        /*size_t reserveSpaceSizeInBytes,*/ reserve_space_size,
        /*void *reserveSpace*/  d_reserve_space), "cudnnMultiHeadAttnForward");

    checkCUDNN(cudnnMultiHeadAttnForward(
        /*cudnnHandle_t handle,*/ cudnn_handle1,
        /*const cudnnAttnDescriptor_t attnDesc,*/ attn_desc1,
        /*int currIdx,*/ -1,
        /*const int loWinIdx[],*/ lo_win_idx,
        /*const int hiWinIdx[],*/ hi_win_idx,
        /*const int devSeqLengthsQO[],*/ d_seq_len_q_o,
        /*const int devSeqLengthsKV[],*/ d_seq_len_k_v,
        /*const cudnnSeqDataDescriptor_t qDesc,*/ seq_data_desc,
        /*const void *queries,*/ q1,
        /*const void *residuals,*/ q1,
        /*const cudnnSeqDataDescriptor_t kDesc,*/ seq_data_desc,
        /*const void *keys,*/ k1,
        /*const cudnnSeqDataDescriptor_t vDesc,*/ seq_data_desc,
        /*const void *values,*/ v1,
        /*const cudnnSeqDataDescriptor_t oDesc,*/ seq_data_desc,
        /*void *out,*/ d_out_seq1,
        /*size_t weightSizeInBytes,*/ weight_size1,
        /*const void *weights,*/ d_weight1,
        /*size_t workSpaceSizeInBytes,*/ workspace_size1,
        /*void *workSpace,*/ d_workspace1,
        /*size_t reserveSpaceSizeInBytes,*/ reserve_space_size1,
        /*void *reserveSpace*/  d_reserve_space1), "cudnnMultiHeadAttnForward1");

    std::cout << "finish !!!" << std::endl;
    return 0;
}

int main() {

    // cudaStream_t stream;
    // checkCUDA(cudaStreamCreate(&stream));

    // cudaStream_t stream1;
    // checkCUDA(cudaStreamCreate(&stream1));

    // func1(stream);
    // func1(stream1);

    func1();
}