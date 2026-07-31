#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

__global__ void saxpy_kernel(float *y, const float *x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main(int argc, char **argv) {
    
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n < 0) {
        fprintf(stderr, "n must be non-negative\n");
        return 1;
    }
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    float *h_x = new float[n];
    float *h_y = new float[n];

    for (int i = 0; i < n; ++i) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    float *d_x, *d_y;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    saxpy_kernel<<<gridSize, blockSize>>>(d_y, d_x, 2.0f, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(h_y[i]);
    }

    printf("SUM=%.0f (n=%d, time=%.3f ms)\n", sum, n, kernel_time_ms);

    delete[] h_x;
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}