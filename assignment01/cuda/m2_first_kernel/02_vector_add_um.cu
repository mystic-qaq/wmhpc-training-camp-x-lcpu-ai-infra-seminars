// 问题 2.3：把显式内存管理改成 Unified Memory（ MODIFY ）。
// 下面是一份完整可运行的显式管理版本。任务：
//   0. 先按原样跑一次，记下耗时——这一版会被你的改动覆盖掉，
//      第 4 步的对比要拿它做基准；(29.8ms -> 33.6ms)
//   1. 用 cudaMallocManaged 替换 cudaMalloc + malloc；
//   2. 删掉所有 cudaMemcpy，kernel 直接读写同一组指针，CPU 也直接读；
//   3. 想清楚哪里需要 cudaDeviceSynchronize；
//   4. 对比两版的耗时。两版的计时窗口要保持一致：分配和填数据都在窗口
//      外，窗口从"数据已经在内存里备好"开始，到 CPU 把结果全部读完为止
//      （下面用一个累加校验和的循环代表"CPU 读完全部结果"，别把它删了）。
// 改完仍要 PASS。
#include <chrono>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;          // 16M 元素
    size_t bytes = (size_t)n * sizeof(float);

    // 建立 CUDA context（初始化开销不计入计时）
    CUDA_CHECK(cudaFree(0));

    // 1. 用 cudaMallocManaged 分配 Unified Memory，同时替代 host 和 device 指针
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));

    // 2. CPU 直接填充数据（在计时窗口外完成）
    fill_random(a, n, 1);
    fill_random(b, n, 2);

    // 期望校验和（也在窗口外计算）
    double want = 0;
    for (int i = 0; i < n; i++) want += (double)(a[i] + b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ================= 计时窗口开始 =================
    auto t0 = std::chrono::steady_clock::now();

    // 3. Kernel 启动，直接使用统一内存指针
    vectorAdd<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK_KERNEL();

    // 4. 必须同步，确保 kernel 完成且 GPU 写入对 CPU 可见
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU 直接读取结果（会触发按需页面迁移，计入计时）
    double got = 0;
    for (int i = 0; i < n; i++) got += (double)c[i];

    auto t1 = std::chrono::steady_clock::now();
    // ================= 计时窗口结束 =================

    printf("Unified Memory 版本: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    REPORT(fabs(got - want) <= 1e-3 * (1.0 + fabs(want)));

    // 可选的释放（不计时）
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    return 0;
}
