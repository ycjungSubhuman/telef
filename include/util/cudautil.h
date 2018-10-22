#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

#ifdef DEBUG
#define CUDA_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)
#else
#define CUDA_CHECK(arg)   arg
#define CHECK_ERROR_MSG(str) do {} while (0)
#endif

inline void __checkErrorFunc(cudaError_t errarg, const char* file,
                             const int line)
{
    if(errarg) {
        fprintf(stderr, "Error \"%s\" at %s(%i)\n", cudaGetErrorName(errarg), file, line);
        exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file,
                              const int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %s(%i): %s\n",
                errstr, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template<typename T>
inline void CUDA_MALLOC(T **dst_d, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dst_d), count*sizeof(T)));
}

template<typename T>
inline void CUDA_ALLOC_AND_COPY(T **dst_d, const T *src_h, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dst_d), count*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(*dst_d, src_h, count*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void CUDA_ALLOC_AND_ZERO(T **dst_d, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dst_d), count*sizeof(T)));
    CUDA_CHECK(cudaMemset(*dst_d, 0, count*sizeof(T)));
}

template<typename T>
inline void CUDA_ZERO(T **dst_d, size_t count) {
    CUDA_CHECK(cudaMemset(*dst_d, 0, count*sizeof(T)));
}

template<typename T>
inline void CUDA_RETRIEVE(T *dst_h, const T *src_d, size_t count) {
    CUDA_CHECK(cudaMemcpy(*dst_h, src_d, count*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void CUDA_FREE(T *p_d) {
    CUDA_CHECK(cudaFree(p_d));
}

inline constexpr unsigned int GET_DIM_GRID(int required_thread, int num_thread) {
    return static_cast<unsigned int>((required_thread + num_thread - 1) / num_thread);
}
