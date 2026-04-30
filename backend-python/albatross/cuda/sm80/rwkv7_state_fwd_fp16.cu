#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <stdio.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>

#ifndef _N_
#define _N_ 64
#endif

typedef at::Half F;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;
#define rotator(_A,_B,_C) (two_to_neg_41*float(ro1*(_A)+ro2*(_B)+ro3*(_C)))
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))

// __global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
//                                 F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                 F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     const int t0 = _elapsed_t[bbb];
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
//     __half state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = static_cast<__half>(_state[j]);
//     __shared__ __half r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
//     for (int _t = 0; _t < T; _t++){
//         const int t = bbb*T*C + h*_N_ + i + _t * C;
//         __syncthreads();
//         r[i] = static_cast<__half>(_r[t]);
//         w[i] = __half(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(t0+_t));
//         k[i] = static_cast<__half>(_k[t]);
//         a[i] = static_cast<__half>(_a[t]);
//         b[i] = static_cast<__half>(_b[t]);
//         __syncthreads();
//         __half sa = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             sa += a[j] * state[j];
//         }
//         __half vv = static_cast<__half>(_v[t]);
//         __half y = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             __half& s = state[j];
//             s += s * w[j] + k[j] * vv + sa * b[j];
//             y += s * r[j];
//         }
//         _y[t] = F(y);
//     }
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = static_cast<F>(state[j]);
// }


__global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);

    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        __half2 sa2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 += a[j] * state[j];
        __half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        __half vv = _v[t];
        __half2 vv2 = {vv, vv};
        __half2 y2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
        {
            __half2 &s = state[j];
            s += s * w[j] + k[j] * vv2 + sa2 * b[j];
            y2 += s * r[j];
        }
        _y[t] = y2.x + y2.y;
    }
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

__global__ void kernel_forward_w0_fp16_dither_one(const int B, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);

    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    const int t = bbb * C + h * _N_ + i;
    // float w0 = __expf(_w[t]);
    // w0 = w0 / (w0 + 1);
    // sigmoid = 1 / (1+exp2f(-nlog2e * x))
    ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((F *)k)[i] = _k[t];
    ((F *)a)[i] = _a[t];
    ((F *)b)[i] = _b[t];
    ((F *)r)[i] = _r[t];
    __syncthreads();
    __half2 sa2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        sa2 += a[j] * state[j];
    __half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};

    __half vv = _v[t];
    __half2 vv2 = {vv, vv};
    __half2 y2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        __half2 &s = state[j];
        s += s * w[j] + k[j] * vv2 + sa2 * b[j];
        y2 += s * r[j];
    }
    _y[t] = y2.x + y2.y;

#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

// __global__ void kernel_forward_w0_fp16_dither_one_ref(const int B, const int C, const int H,
//                                 F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                 F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
//     F state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = _state[j];
//     __shared__ F r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
//     const int t = bbb*C + h*_N_ + i;
//     __syncthreads();
//     r[i] = _r[t];
//     w[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]));
//     k[i] = _k[t];
//     a[i] = _a[t];
//     b[i] = _b[t];
//     __syncthreads();
//     F sa = 0;
//     #pragma unroll
//     for (int j = 0; j < _N_; j++){
//         sa += a[j] * state[j];
//     }
//     F vv = F(_v[t]);
//     F y = 0;
//     #pragma unroll
//     for (int j = 0; j < _N_; j++){
//         F& s = state[j];
//         s += s * w[j] + k[j] * vv + sa * b[j];
//         y += s * r[j];
//     }
//     _y[t] = F(y);
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = state[j];
// }

void cuda_forward_seq(int B, int T, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    // auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_seq<<<B * H, _N_>>>(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_forward_one(int B, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_one<<<B * H, _N_, 0, stream>>>(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

// Fused sampling kernels ported from rwkv_lightning_libtorch for Albatross sampler dispatch.

#include <iostream>
#include <math.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>
typedef curandStatePhilox4_32_10_t RAND;

template <typename T, typename ReduceOp>
__device__ __forceinline__ void warpReduceAll(T& val, ReduceOp op) {
    #pragma unroll
    for (int offset = 16; offset>0; offset/=2) {
        val = op(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
}

template <typename T, typename ReduceOp, int BLOCK_SIZE=1024, bool monotone_sum=false>
__device__ __forceinline__ void blockReduceAll(T& val, ReduceOp op, T identity, void* buf) {
    T* warpResults = reinterpret_cast<T*>(buf);
    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    const int numWarps = (BLOCK_SIZE + 31) / 32;
    warpReduceAll(val, op);
    if (lane == 31) warpResults[warpId] = val;
    __syncthreads();
    T warpVal;
    if constexpr (!monotone_sum) {
        warpVal = (threadIdx.x < numWarps) ? warpResults[threadIdx.x] : identity;
        if (threadIdx.x < 32) warpReduceAll(warpVal, op);
        if (threadIdx.x == 0) warpResults[0] = warpVal;
    }
    else {
        if (threadIdx.x == 0){
            warpVal = warpResults[0];
            #pragma unroll
            for (int i=1; i<numWarps; i++){
                warpVal += warpResults[i];
            }
            warpResults[0] = warpVal;
        }
    }
    __syncthreads();
    val = warpResults[0];
}

template <typename T>
__device__ __forceinline__ T warpInclusiveScan(T val) {
    #pragma unroll
    for (int offset = 16; offset>0; offset/=2) {
        T n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += n;
        }
    }
    return val;
}

// Block-level inclusive scan - each thread gets sum of itself and all preceding threads
template <typename T, int BLOCK_SIZE = 1024>
__device__ __forceinline__ T blockInclusiveScan(T val, void* buf /* shared */ , void* total=nullptr) {
    T* warpSums = reinterpret_cast<T*>(buf);

    const int lane = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    constexpr int numWarps = (BLOCK_SIZE + 31) / 32;

    // Step 1: Inclusive scan within each warp (ok)
    T val1 = warpInclusiveScan(val);

    // Step 2: Last lane of each warp stores its total
    if (lane == 31) {
        warpSums[warpId] = val1;
    }
    __syncthreads();

    // Step 3: First warp does inclusive scan of warp totals
    // if (threadIdx.x < numWarps) {
    //     T warpTotal = warpSums[threadIdx.x];
    //     warpTotal = warpInclusiveScan(warpTotal);
    //     warpSums[threadIdx.x] = warpTotal;
    // }
    // MUST sum this way to ensure numerical MONOTONICITY (not STABILITY)
    if (threadIdx.x == 0){
        T s = warpSums[0];
        #pragma unroll
        for (int i=1; i<numWarps; i++){
            s += warpSums[i];
            warpSums[i] = s;
        }
    }
    __syncthreads();

    // Step 4: Add previous warp's prefix to current value
    if (warpId > 0) {
        val1 += warpSums[warpId - 1];
    }
    if (threadIdx.x == BLOCK_SIZE-1 && total != nullptr) {
        *reinterpret_cast<T*>(total) = val1;
    }
    __syncthreads();
    return val1;
}

// Reduction operation functors
template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
    static constexpr T identity() { return T(0); }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return max(a, b); }
    static constexpr T identity() { return -INFINITY; }  // For float
};

template <typename T>
struct MinOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return min(a, b); }
    static constexpr T identity() { return INFINITY; }   // For float
};

template <typename T>
struct ProdOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
    static constexpr T identity() { return T(1); }
};

__device__ __forceinline__ float sf(float x){
    float y = isnan(x) ? 0.0f : x;
    return (isinf(y) ? copysignf(FLT_MAX, y) : y);
}

__global__ void setup_rand_kernel(RAND* states, unsigned long long seed) {
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

at::Tensor setup_rand(int64_t seed, int64_t B){
    at::Tensor state = at::zeros({(long)(B*sizeof(RAND))}, at::TensorOptions().dtype(at::kChar).device(at::kCUDA));
    setup_rand_kernel<<<((int)B), 1>>>((RAND*)(state.data_ptr()), (unsigned long long)seed);
    return state;
}

// #define P0i(x) do{printf(#x":%d\n",x);}while(0)
// #define P0f(x) do{printf(#x":%8e\n",x);}while(0)

__device__ __forceinline__ void print_bits_u32(unsigned v)
{
    for (int bit = 0; bit < 32; ++bit) {
        if(bit%8==0) printf(" ");
        printf("%d", int(bool(v & (1<<bit))));
    }
    printf("\n");
}

__device__ __forceinline__ void dump_thread_states(const unsigned int* s_state, int nthreads)
{
    __syncthreads();
    if (threadIdx.x != 0) return;

    for (int i = 0; i < nthreads; ++i) {
        printf("%3d:", i*32);
        print_bits_u32(s_state[i]);
    }
    printf("\n");
}


#define BLOCKDIM_X_SAMPLE 1024
__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1) batch_sampling_repetition_temperature_topk_topp_kernel(
    const int B,
    const int T,                         // should be 1 typically; may not be 1 if full output is obtained
    const int V,                         // vocabulary size, 60,000 ~ 120,000
    const float *__restrict__ logits,    // (B, V) if T == 1; If T != 1, only logits[:, T-1, :] is read. This avoids another copying operation
          float *__restrict__ penalties, // (B, V), can set some to -INF for masking
          int   *__restrict__ outputs,   // (B,)
          RAND  *__restrict__ states,    // random state, typedef curandStatePhilox4_32_10_t RAND;
          float *__restrict__ probs,     // probs (in L2 cache)
    const float presence_penalty,
    const float repetition_penalty,
    const float penalty_decay,
    const float log2_inv_temp,
    const int   top_k,
    const float top_p
) {
    const int b = blockIdx.x;
    const int d = blockDim.x;
    const int t = threadIdx.x;
    const int w = t / 32;
    const int l = t % 32;
    // constexpr int W = (BLOCKDIM_X_SAMPLE + 31) / 32;
    __shared__ __align__(256) char reduce_buf[256];
    __builtin_assume(BLOCKDIM_X_SAMPLE == d);
    __builtin_assume(V % 4 == 0);
    __builtin_assume(V <= 1048576);
    __builtin_assume(log2_inv_temp > 0.f);
    const int V4 = V / 4;
    float4 l4, p4;

    logits    += (b*T+(T-1)) * V;  // B T V
    penalties +=           b * V;  // B V
    outputs   +=           b    ;  // B
    states    +=           b    ;  // B
    probs     += (b*T+(T-1)) * V;  // B T V

    float maxu = -INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)logits)[i];
        p4 = ((float4*)penalties)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fl = ((float*)&l4)[j];
            // if (i*4+j < 3){
            //     P0i(i*4+j);
            //     P0f(fl);
            // }
            float &fp = ((float*)&p4)[j];
            fl = sf((sf(fl)-fp) * log2_inv_temp);
            maxu = max(maxu, fl);
            // ((float*)&l4)[j] = fr;
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    float exp_denom = 0;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        float em = 0.f;
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            em += exp2f(fr-maxu);
        }
        exp_denom += em;
    }
    blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    float pmax = -INFINITY;
    float pmin = +INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            fr = exp2f(fr-maxu) / exp_denom;
            pmax = max(pmax, fr);
            pmin = min(pmin, fr);
            // ((float*)&l4)[j] = fr;
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
    __syncthreads();

    // if(t==0) P0f(pmax);
    unsigned left =  __float_as_uint(pmin), right =  __float_as_uint(pmax) + 1;

    uint4 cnt = {(unsigned)V, 0, 0, 0};
    l4 = {1, 0, 0, 0};
    uint4 pivot;
    while ((cnt.x > top_k || l4.x > top_p) && left < right-1) {
        // if(t==0){
        //     P0i(top_k);
        //     P0i(left);
        //     P0i(right);
        //     P0i(cnt.x);
        //     printf("\n");
        // }
        pivot.x = left;
        pivot.z = (left            + right) / 2;
        pivot.y = (left  + pivot.z        ) / 2;
        pivot.w = (        pivot.z + right) / 2;
        l4.y = l4.z = l4.w = 0;
        cnt.y = cnt.z = cnt.w = 0;
        for (int i=t; i<V4; i+=d) {
            p4 = ((float4*)probs)[i];
            #pragma unroll
            for (int j=0; j<4; j++){
                float &p = ((float*)&p4)[j];
                if (p >= __uint_as_float(pivot.y)) {cnt.y ++; l4.y += p;}
                if (p >= __uint_as_float(pivot.z)) {cnt.z ++; l4.z += p;}
                if (p >= __uint_as_float(pivot.w)) {cnt.w ++; l4.w += p;}
            }
        }
        blockReduceAll(cnt.y, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.y < top_k && l4.y < top_p){
            left = pivot.x;
            right = pivot.y;
            // cnt.x = cnt.x;
            // l4.x = l4.x;
            continue;
        }
        blockReduceAll(cnt.z, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.z < top_k && l4.z < top_p){
            left = pivot.y;
            right = pivot.z;
            cnt.x = cnt.y;
            l4.x = l4.y;
            continue;
        }
        blockReduceAll(cnt.w, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.w < top_k && l4.w < top_p){
            left = pivot.z;
            right = pivot.w;
            cnt.x = cnt.z;
            l4.x = l4.z;
            continue;
        }
        left = pivot.w;
        // right = right;
        cnt.x = cnt.w;
        l4.x = l4.w;
    }
    // return left
    float threshold =  __uint_as_float(left);
    // if(t==0) P0f(threshold);
    // 5. recompute (read once)
    float gtp=0;
    unsigned eqk=0, gtk=0;
    __shared__ float /* seqp, */ sgtp;
    __shared__ unsigned seqk, sgtk;

    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            if (p == threshold) eqk ++;
            if (p >  threshold) {gtk ++; gtp += p;}
        }
    }
    // s: shared all
    // c: cumulative
    // -: per thread
    // __syncthreads();
    float    cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
    __syncthreads();
    unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
    __syncthreads();
    unsigned cgtk = blockInclusiveScan(gtk, reduce_buf, &sgtk);
    __syncthreads();
    // if(t==0) P0f(sgtp);
    // if(t==0) P0i(seqk);
    // if(t==0) P0i(sgtk);

    // compute compensation
    // seqk == total number of tokens that equals threshold
    // _gtp + threshold * _eqk == _eqp
    // (top_p - sgtp) == delta_p
    // delta_p / seqp
    unsigned neqk = seqk;
    float comp=1.0f;
    if (neqk > 0){
        comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
        comp = min(sf(float(top_k - sgtk) / neqk), comp);
        comp = max(comp, 0.0f);
    }

    // 6. Yield sampled tokens
    __shared__ float randp, sum_p;
    __shared__ float4 rand4;
    __shared__ int idxt;
    float actual_p = gtp + (threshold * eqk) * comp;
    __syncthreads();
    float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
    __syncthreads();
    if (t==0){
        idxt = 0;
        rand4 = curand_uniform4(states);
        randp = sum_p * rand4.x; // only once
    }
    __syncthreads();

    bool u = (randp <= cumu_p);
    // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    __syncthreads();
    bool u_ = __shfl_up_sync(0xffffffff, u, 1);
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    if (u!=u_) idxt=t;
    __syncthreads();

    // a sub-tile (of no more than 1024)
    int idn = idxt*4 + (t/4)*4*d + (t%4);
    // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
    float o0 = (idn<V) ? (probs[idn]) : 0;
    float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

    __shared__ float sum_o;
    float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o); // monotone
    __syncthreads();
    float rand_2 = sum_o * rand4.y;
    u = (rand_2 <= cumu_o);
    // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    // u: current u_: prev
    // at first thread: u_ == 0
    u_ = __shfl_up_sync(0xffffffff, u, 1);
    __syncthreads();
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    // write idn
    __shared__ int out_id;
    if (u!=u_) out_id = (idn<V)? idn: 0;
    __syncthreads();
    idn = out_id;
    if (t==0) *outputs = idn;
    // 7. Update penalties
    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)penalties)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            int idp = i*4+j;
            p = fmaf(p, penalty_decay, ((idn != idp) ? 0 : (p == 0 ? presence_penalty : repetition_penalty)));
        }
        ((float4*)penalties)[i] = p4;
    }
}

at::Tensor batch_sampling_repetition_temperature_topk_topp(
    at::Tensor& logits,
    at::Tensor& penalties,
    at::Tensor& states,
    double      presence_penalty,
    double      repetition_penalty,
    double      penalty_decay,
    double      temperature,
    int64_t     top_k,
    double      top_p
) {
    int B, T, V;
    if (logits.dtype() != at::kFloat) {
        throw std::invalid_argument(
            "Logits tensor must be of type float32 (FP32), got " +
            std::string(logits.dtype().name()) + " !\n"
        );
    }
    V = logits.size(-1);
    B = (penalties.dim() == 2)? penalties.size(0): 1;
    T = (logits.dim() == 3)? logits.size(1): 1;
    if (!(V > 0 && V <= 1048576 && V % 4 == 0)){
        throw std::invalid_argument(
            "Vocabulary size must be multiple of 4, and no larger than 1048576, got " + std::to_string(V)
            + " !\n"
        );
    }
    if (!(B > 0 && T > 0)){
        throw std::invalid_argument(
            "B and T must be positive, got B=" + std::to_string(B)
            + ", T=" + std::to_string(T)
            + " !\n"
        );
    }
    if (!(temperature >= 0.001 && temperature <= 1000)){
        throw std::invalid_argument(
            "Temperature outside range, got " + std::to_string(temperature)
            + ", expect [0.001, 1000]!\n"
        );
    }
    if (top_k <= 0 || top_k > V) top_k = V;
    if (top_p < 0 || top_p > 1)  top_p = 1;
    if (top_p == 0) {
        top_k = 1;
        top_p = 1;
    }
    double log2e_inv_temp = M_LOG2E / temperature;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto probs = at::empty({B,V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    if (B*V*4 <= 4194304) {
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = probs.data_ptr();
        stream_attribute.accessPolicyWindow.num_bytes = B*V*4;
        stream_attribute.accessPolicyWindow.hitRatio  = 1;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    }
    auto out = at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    batch_sampling_repetition_temperature_topk_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V,
        (float*)logits.data_ptr(),
        (float*)penalties.data_ptr(),
        (int*)  out.data_ptr(),
        (RAND*) states.data_ptr(),
        (float*)probs.data_ptr(),
        (float) presence_penalty,
        (float) repetition_penalty,
        (float) penalty_decay,
        (float) log2e_inv_temp,
        (int)   top_k,
        (float) top_p
    );
    return out;
}

__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1) batch_sampling_temperature_topk_topp_kernel(
    const int B,
    const int T,                         // should be 1 typically; may not be 1 if full output is obtained
    const int V,                         // vocabulary size, 60,000 ~ 120,000
    const float *__restrict__ logits,    // (B, V) if T == 1; If T != 1, only logits[:, T-1, :] is read. This avoids another copying operation
          int   *__restrict__ outputs,   // (B,)
          RAND  *__restrict__ states,    // random state, typedef curandStatePhilox4_32_10_t RAND;
          float *__restrict__ probs,     // probs (in L2 cache)
    const float log2e_inv_temp,
    const int   top_k,
    const float top_p
) {
    const int b = blockIdx.x;
    const int d = blockDim.x;
    const int t = threadIdx.x;
    const int w = t / 32;
    const int l = t % 32;
    __shared__ __align__(256) char reduce_buf[256];
    __builtin_assume(BLOCKDIM_X_SAMPLE == d);
    __builtin_assume(V % 4 == 0);
    __builtin_assume(V <= 1048576);
    __builtin_assume(log2e_inv_temp > 0.f);
    const int V4 = V / 4;
    float4 l4, p4;

    logits    += (b*T+(T-1)) * V;  // B T V
    outputs   +=           b    ;  // B
    states    +=           b    ;  // B
    probs     += (b*T+(T-1)) * V;  // B T V

    float maxu = -INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)logits)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fl = ((float*)&l4)[j];
            fl = sf(fl * log2e_inv_temp);
            maxu = max(maxu, fl);
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    float exp_denom = 0;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        float em = 0.f;
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            em += exp2f(fr-maxu);
        }
        exp_denom += em;
    }
    blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    float pmax = -INFINITY;
    float pmin = +INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            fr = exp2f(fr-maxu) / exp_denom;
            pmax = max(pmax, fr);
            pmin = min(pmin, fr);
            // ((float*)&l4)[j] = fr;
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
    __syncthreads();

    // if(t==0) P0f(pmax);
    unsigned left =  __float_as_uint(pmin), right =  __float_as_uint(pmax) + 1;

    uint4 cnt = {(unsigned)V, 0, 0, 0};
    l4 = {1, 0, 0, 0};
    uint4 pivot;
    while ((cnt.x > top_k || l4.x > top_p) && left < right-1) {
        pivot.x = left;
        pivot.z = (left            + right) / 2;
        pivot.y = (left  + pivot.z        ) / 2;
        pivot.w = (        pivot.z + right) / 2;
        l4.y = l4.z = l4.w = 0;
        cnt.y = cnt.z = cnt.w = 0;
        for (int i=t; i<V4; i+=d) {
            p4 = ((float4*)probs)[i];
            #pragma unroll
            for (int j=0; j<4; j++){
                float &p = ((float*)&p4)[j];
                if (p >= __uint_as_float(pivot.y)) {cnt.y ++; l4.y += p;}
                if (p >= __uint_as_float(pivot.z)) {cnt.z ++; l4.z += p;}
                if (p >= __uint_as_float(pivot.w)) {cnt.w ++; l4.w += p;}
            }
        }
        blockReduceAll(cnt.y, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.y < top_k && l4.y < top_p){
            left = pivot.x;
            right = pivot.y;
            // cnt.x = cnt.x;
            // l4.x = l4.x;
            continue;
        }
        blockReduceAll(cnt.z, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.z < top_k && l4.z < top_p){
            left = pivot.y;
            right = pivot.z;
            cnt.x = cnt.y;
            l4.x = l4.y;
            continue;
        }
        blockReduceAll(cnt.w, SumOp<unsigned>{}, SumOp<unsigned>::identity(), reduce_buf);
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (cnt.w < top_k && l4.w < top_p){
            left = pivot.z;
            right = pivot.w;
            cnt.x = cnt.z;
            l4.x = l4.z;
            continue;
        }
        left = pivot.w;
        // right = right;
        cnt.x = cnt.w;
        l4.x = l4.w;
    }
    // return left
    float threshold =  __uint_as_float(left);
    // if(t==0) P0f(threshold);
    // 5. recompute (read once)
    float gtp=0;
    unsigned eqk=0, gtk=0;
    __shared__ float /* seqp, */ sgtp;
    __shared__ unsigned seqk, sgtk;

    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            if (p == threshold) eqk ++;
            if (p >  threshold) {gtk ++; gtp += p;}
        }
    }
    // s: shared all
    // c: cumulative
    // -: per thread
    // __syncthreads();
    float    cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
    __syncthreads();
    unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
    __syncthreads();
    unsigned cgtk = blockInclusiveScan(gtk, reduce_buf, &sgtk);
    __syncthreads();
    // if(t==0) P0f(sgtp);
    // if(t==0) P0i(seqk);
    // if(t==0) P0i(sgtk);

    // compute compensation
    // seqk == total number of tokens that equals threshold
    // _gtp + threshold * _eqk == _eqp
    // (top_p - sgtp) == delta_p
    // delta_p / seqp
    unsigned neqk = seqk;
    float comp=1.0f;
    if (neqk > 0){
        comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
        comp = min(sf(float(top_k - sgtk) / neqk), comp);
        comp = max(comp, 0.0f);
    }

    // 6. Yield sampled tokens
    __shared__ float randp, sum_p;
    __shared__ float4 rand4;
    __shared__ int idxt;
    float actual_p = gtp + (threshold * eqk) * comp;
    __syncthreads();
    float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
    __syncthreads();
    if (t==0){
        idxt = 0;
        rand4 = curand_uniform4(states);
        randp = sum_p * rand4.x; // only once
    }
    __syncthreads();

    bool u = (randp <= cumu_p);
    // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    __syncthreads();
    bool u_ = __shfl_up_sync(0xffffffff, u, 1);
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    if (u!=u_) idxt=t;
    __syncthreads();

    // a sub-tile (of no more than 1024)
    int idn = idxt*4 + (t/4)*4*d + (t%4);
    // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
    float o0 = (idn<V) ? (probs[idn]) : 0;
    float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

    __shared__ float sum_o;
    float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o); // monotone
    __syncthreads();
    float rand_2 = sum_o * rand4.y;
    u = (rand_2 <= cumu_o);
    // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    // u: current u_: prev
    // at first thread: u_ == 0
    u_ = __shfl_up_sync(0xffffffff, u, 1);
    __syncthreads();
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    // write idn
    __shared__ int out_id;
    if (u!=u_) out_id = (idn<V)? idn: 0;
    __syncthreads();
    idn = out_id;
    if (t==0) *outputs = idn;
}




__global__ void __launch_bounds__(BLOCKDIM_X_SAMPLE, 1) batch_sampling_topp_kernel(
    const int B,
    const int T,                         // should be 1 typically; may not be 1 if full output is obtained
    const int V,                         // vocabulary size, 60,000 ~ 120,000
    const float *__restrict__ logits,    // (B, V) if T == 1; If T != 1, only logits[:, T-1, :] is read. This avoids another copying operation
          int   *__restrict__ outputs,   // (B,)
          RAND  *__restrict__ states,    // random state, typedef curandStatePhilox4_32_10_t RAND;
          float *__restrict__ probs,     // probs (in L2 cache)
    const float top_p
) {
    const int b = blockIdx.x;
    const int d = blockDim.x;
    const int t = threadIdx.x;
    const int w = t / 32;
    const int l = t % 32;
    __shared__ __align__(256) char reduce_buf[256];
    __builtin_assume(BLOCKDIM_X_SAMPLE == d);
    __builtin_assume(V % 4 == 0);
    __builtin_assume(V <= 1048576);
    const int V4 = V / 4;
    float4 l4, p4;

    logits    += (b*T+(T-1)) * V;  // B T V
    outputs   +=           b    ;  // B
    states    +=           b    ;  // B
    probs     += (b*T+(T-1)) * V;  // B T V

    float maxu = -INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)logits)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fl = ((float*)&l4)[j];
            fl = sf(fl * float(M_LOG2E));
            maxu = max(maxu, fl);
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(maxu, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    float exp_denom = 0;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        float em = 0.f;
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            em += exp2f(fr-maxu);
        }
        exp_denom += em;
    }
    blockReduceAll(exp_denom, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
    __syncthreads();
    float pmax = -INFINITY;
    float pmin = +INFINITY;
    for (int i=t; i<V4; i+=d) {
        l4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &fr = ((float*)&l4)[j];
            fr = exp2f(fr-maxu) / exp_denom;
            pmax = max(pmax, fr);
            pmin = min(pmin, fr);
        }
        ((float4*)probs)[i] = l4;
    }
    blockReduceAll(pmax, MaxOp<float>{}, MaxOp<float>::identity(), reduce_buf);
    __syncthreads();
    blockReduceAll(pmin, MinOp<float>{}, MinOp<float>::identity(), reduce_buf);
    __syncthreads();

    // if(t==0) P0f(pmax);
    unsigned left =  __float_as_uint(pmin), right =  __float_as_uint(pmax) + 1;

    // uint4 cnt = {(unsigned)V, 0, 0, 0};
    l4 = {1, 0, 0, 0};
    uint4 pivot;
    while ((l4.x > top_p) && left < right-1) {
        pivot.x = left;
        pivot.z = (left            + right) / 2;
        pivot.y = (left  + pivot.z        ) / 2;
        pivot.w = (        pivot.z + right) / 2;
        l4.y = l4.z = l4.w = 0;
        for (int i=t; i<V4; i+=d) {
            p4 = ((float4*)probs)[i];
            #pragma unroll
            for (int j=0; j<4; j++){
                float &p = ((float*)&p4)[j];
                if (p >= __uint_as_float(pivot.y)) l4.y += p;
                if (p >= __uint_as_float(pivot.z)) l4.z += p;
                if (p >= __uint_as_float(pivot.w)) l4.w += p;
            }
        }
        __syncthreads();
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.y, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (l4.y < top_p){
            left = pivot.x;
            right = pivot.y;
            continue;
        }
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.z, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (l4.z < top_p){
            left = pivot.y;
            right = pivot.z;
            l4.x = l4.y;
            continue;
        }
        blockReduceAll<float, SumOp<float>, BLOCKDIM_X_SAMPLE, true>(l4.w, SumOp<float>{}, SumOp<float>::identity(), reduce_buf);
        __syncthreads();
        if (l4.w < top_p){
            left = pivot.z;
            right = pivot.w;
            l4.x = l4.z;
            continue;
        }
        left = pivot.w;
        l4.x = l4.w;
    }
    // return left
    float threshold =  __uint_as_float(left);
    // if(t==0) P0f(threshold);
    // 5. recompute (read once)
    float gtp=0;
    unsigned eqk=0;
    __shared__ float /* seqp, */ sgtp;
    __shared__ unsigned seqk;

    for (int i=t; i<V4; i+=d) {
        p4 = ((float4*)probs)[i];
        #pragma unroll
        for (int j=0; j<4; j++){
            float &p = ((float*)&p4)[j];
            // bool u0 = (p == threshold);
            // bool u1 = (p > threshold);
            // eqk += u0;
            // gtp = fmaf(p, u1, gtp);
            if (p == threshold) eqk ++;
            if (p >  threshold) gtp += p;
        }
    }
    float    cgtp = blockInclusiveScan(gtp, reduce_buf, &sgtp);
    __syncthreads();
    unsigned ceqk = blockInclusiveScan(eqk, reduce_buf, &seqk);
    __syncthreads();
    unsigned neqk = seqk;
    float comp=1.0f;
    if (neqk > 0){
        comp = min(sf((top_p - sgtp) / (threshold * neqk)), comp);
        comp = max(comp, 0.0f);
    }

    // 6. Yield sampled tokens
    __shared__ float randp, sum_p;
    __shared__ float4 rand4;
    __shared__ int idxt;
    float actual_p = gtp + (threshold * eqk) * comp;
    __syncthreads();
    float cumu_p = blockInclusiveScan(actual_p, reduce_buf, &sum_p);
    __syncthreads();
    if (t==0){
        idxt = 0;
        rand4 = curand_uniform4(states);
        randp = sum_p * rand4.x; // only once
    }
    __syncthreads();

    bool u = (randp <= cumu_p);
    // at last thread: randp = sum_p * rand4.x < cumu_p == sum_p, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    __syncthreads();
    bool u_ = __shfl_up_sync(0xffffffff, u, 1);
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    if (u!=u_) idxt=t;
    __syncthreads();

    // a sub-tile (of no more than 1024)
    int idn = idxt*4 + (t/4)*4*d + (t%4);
    // .... .... (idxt) |||| .... .... .... |||| .... .... .... |||| ....
    float o0 = (idn<V) ? (probs[idn]) : 0;
    float o = (o0 < threshold) ? 0 : (o0 == threshold) ? (o0 * comp) : o0;

    __shared__ float sum_o;
    float cumu_o = blockInclusiveScan(o, reduce_buf, &sum_o); // monotone
    __syncthreads();
    float rand_2 = sum_o * rand4.y;
    u = (rand_2 <= cumu_o);
    // at last thread: cumu_o == sum_o, rand4.y < 1, sum_o * rand4.y < cumu_o, u == 1
    if(l==31) ((unsigned*)reduce_buf)[w] = u;
    // u: current u_: prev
    // at first thread: u_ == 0
    u_ = __shfl_up_sync(0xffffffff, u, 1);
    __syncthreads();
    if(t==0) u_=0;
    else if(l==0) u_ = ((unsigned*)reduce_buf)[w-1];
    __syncthreads();

    // write idn
    __shared__ int out_id;
    if (u!=u_) out_id = (idn<V)? idn: 0;
    __syncthreads();
    idn = out_id;
    if (t==0) *outputs = idn;
}



at::Tensor batch_sampling_temperature_topk_topp(
    at::Tensor& logits,
    at::Tensor& states,
    double      temperature,
    int64_t     top_k,
    double      top_p
) {
    int B, T, V;
    if (logits.dtype() != at::kFloat) {
        throw std::invalid_argument(
            "Logits tensor must be of type float32 (FP32), got " +
            std::string(logits.dtype().name()) + " !\n"
        );
    }
    V = logits.size(-1);
    B = (logits.dim() >= 2)? logits.size(0): 1;
    T = (logits.dim() == 3)? logits.size(1): 1;

    if (!(V > 0 && V <= 1048576 && V % 4 == 0)){
        throw std::invalid_argument(
            "Vocabulary size must be multiple of 4, and no larger than 1048576, got " + std::to_string(V)
            + " !\n"
        );
    }
    if (!(B > 0 && T > 0)){
        throw std::invalid_argument(
            "B and T must be positive, got B=" + std::to_string(B)
            + ", T=" + std::to_string(T)
            + " !\n"
        );
    }
    if (!(temperature >= 0.001 && temperature <= 1000)){
        throw std::invalid_argument(
            "Temperature outside range, got " + std::to_string(temperature)
            + ", expect [0.001, 1000]!\n"
        );
    }
    if (top_k <= 0 || top_k > V) top_k = V;
    if (top_p < 0 || top_p > 1)  top_p = 1;
    if (top_p == 0) {
        top_k = 1;
        top_p = 1;
    }
    double log2e_inv_temp = M_LOG2E / temperature;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto probs = at::empty({B,V}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    if (B*V*4 <= 4194304) {
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = probs.data_ptr();
        stream_attribute.accessPolicyWindow.num_bytes = B*V*4;
        stream_attribute.accessPolicyWindow.hitRatio  = 1;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    }
    auto out = at::empty({B}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    if(temperature == 1 && top_k == V) batch_sampling_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V,
        (float*)logits.data_ptr(),
        (int*)  out.data_ptr(),
        (RAND*) states.data_ptr(),
        (float*)probs.data_ptr(),
        (float) top_p
    );
    else batch_sampling_temperature_topk_topp_kernel<<<B, 1024, 0, stream>>>(
        B, T, V,
        (float*)logits.data_ptr(),
        (int*)  out.data_ptr(),
        (RAND*) states.data_ptr(),
        (float*)probs.data_ptr(),
        (float) log2e_inv_temp,
        (int)   top_k,
        (float) top_p
    );
    return out;
}
