//Question 1



/*#include <stdio.h>
#include <cuda_runtime.h>




#define N 1024
#define THREADS_PER_BLOCK 256

// ---------------- Kernel: C[i] = A[i] + B[i] ----------------
__global__ void addArrays(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int* h_A, * h_B, * h_C;      // Host arrays (pointers)
    int* d_A, * d_B, * d_C;      // Device arrays
    int size = N * sizeof(int);

    // ---------------- Allocate Host Memory ----------------
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // ---------------- Initialize Host Arrays ----------------
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // ---------------- Allocate Device Memory ----------------
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // ---------------- Copy Host → Device ----------------
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ---------------- Launch Kernel ----------------
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    addArrays << <blocks, THREADS_PER_BLOCK >> > (d_A, d_B, d_C, N);

    cudaDeviceSynchronize(); // wait for kernel to finish

    // ---------------- Copy Device → Host ----------------
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // ---------------- Verify Result ----------------
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d: %d + %d != %d\n", i, h_A[i], h_B[i], h_C[i]);
            return -1;
        }
        else{
            printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
		}
    }
    printf("Success! All results are correct.\n");

    // ---------------- Cleanup ----------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}*/









//Question 2


/*#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

// ---------------- Kernel 1: C[i] = A[i] + B[i] ----------------
__global__ void kernel1(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------- Kernel 2: D[i] = C[i] * C[i] ----------------
__global__ void kernel2(int* C, int* D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = C[idx] * C[idx];
    }
}

int main() {
    int* h_A, * h_B, * h_C, * h_D;  // Host arrays
    int* d_A, * d_B, * d_C, * d_D;  // Device arrays
    int size = N * sizeof(int);

    // ---------------- Allocate Host Memory ----------------
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_D = (int*)malloc(size);

    // ---------------- Initialize Host Arrays ----------------
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // ---------------- Allocate Device Memory ----------------
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    // ---------------- Copy Host → Device ----------------
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ---------------- Kernel Launches (serial on default stream) ----------------
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    kernel1 << <blocks, THREADS_PER_BLOCK >> > (d_A, d_B, d_C, N);
    kernel2 << <blocks, THREADS_PER_BLOCK >> > (d_C, d_D, N);

    cudaDeviceSynchronize(); // wait for both kernels to finish

    // ---------------- Copy Device → Host ----------------
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // ---------------- Verify & Print ----------------
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("A[%d]=%d, B[%d]=%d → C[%d]=%d → D[%d]=%d\n",
            i, h_A[i], i, h_B[i], i, h_C[i], i, h_D[i]);
    }

    // ---------------- Cleanup ----------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}
*/










//Question 3


/*#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

// ---------------- Kernel 1: C[i] = A[i] + B[i] ----------------
__global__ void kernel1(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------- Kernel 2: D[i] = C[i] * C[i] ----------------
__global__ void kernel2(int* C, int* D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        D[idx] = C[idx] * C[idx];
    }
}

int main() {
    int* h_A, * h_B, * h_C, * h_D;  // Host arrays
    int* d_A, * d_B, * d_C, * d_D;  // Device arrays
    int size = N * sizeof(int);

    // ---------------- Allocate Host Memory ----------------
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_D = (int*)malloc(size);

    // ---------------- Initialize Host Arrays ----------------
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // ---------------- Allocate Device Memory ----------------
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    // ---------------- Copy Host → Device ----------------
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ---------------- Create Streams ----------------
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // ---------------- Launch Kernels on Different Streams ----------------
    kernel1 << <blocks, THREADS_PER_BLOCK, 0, stream1 >> > (d_A, d_B, d_C, N);
    kernel2 << <blocks, THREADS_PER_BLOCK, 0, stream2 >> > (d_C, d_D, N);

    // Wait for both streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ---------------- Copy Device → Host ----------------
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // ---------------- Verify & Print ----------------
    printf("First 10 results:\n");
    for (int i = 0; i < N; i++) {
        printf("A[%d]=%d, B[%d]=%d → C[%d]=%d → D[%d]=%d\n",
            i, h_A[i], i, h_B[i], i, h_C[i], i, h_D[i]);
    }

    // ---------------- Cleanup ----------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
*/










//Question 4

/*#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

// Kernel 1: Computes C[i] = A[i] + B[i]
__global__ void addArrays(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Kernel 2: Computes D[i] = C[i] * C[i]
__global__ void squareArray(int* C, int* D, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        D[i] = C[i] * C[i];
    }
}

int main() {
    // === 1. SETUP ===
    int size = N * sizeof(int);

    // Host (CPU) arrays
    int* h_A, * h_B, * h_D;
    // Device (GPU) arrays
    int* d_A, * d_B, * d_C, * d_D;

    // Allocate memory on the CPU
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_D = (int*)malloc(size);

    // Initialize CPU data
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 10;
    }

    // Allocate memory on the GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_D, size);

    // === 2. EXECUTION ===
    // Copy input data from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks needed
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernels on the GPU's default stream.
    // Tasks on the default stream run in the order they are launched.
    addArrays<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    squareArray<<<blocks, THREADS_PER_BLOCK>>>(d_C, d_D, N);

    // Wait for the GPU to finish all work.
    // This is CRITICAL before copying results back.
    cudaDeviceSynchronize();

    // Copy the final result from GPU back to CPU
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // === 3. VERIFICATION ===
    printf("Verifying first 10 results:\n");
    for (int i = 0; i < 10; i++) {
        int expected = (h_A[i] + h_B[i]) * (h_A[i] + h_B[i]);
        printf("Index %d:  Result = %-5d  |  Expected = %d\n", i, h_D[i], expected);
    }
    printf("...\n");

    // === 4. CLEANUP ===
    free(h_A);
    free(h_B);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}*/









//Question 5


/*#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

__global__ void addArrays(int* A, int* B, int* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void print_results(int* arr, int* expected_a, int* expected_b) {
    for (int i = 0; i < 5; i++) {
        printf("Index %d: %d + %d = %d\n", i, expected_a[i], expected_b[i], arr[i]);
    }
    printf("...\n");
    int last_idx = N - 1;
    printf("Index %d: %d + %d = %d\n", last_idx, expected_a[last_idx], expected_b[last_idx], arr[last_idx]);
}


int main() {
    int size = N * sizeof(int);

    int* h_A, * h_B, * h_C;
    int* d_A, * d_B, * d_C;

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 10;
        h_B[i] = i * 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    printf("--- Case 1: Launching <<<1, %d>>> ---\n", N);

    dim3 blocks_case1(1);
    dim3 threads_case1(N);

    addArrays << <blocks_case1, threads_case1 >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    print_results(h_C, h_A, h_B);
    printf("\n");

    printf("--- Case 2: Launching <<<%d, 32>>> ---\n", N / 32);

    int threadsPerBlock = 32;
    int blocksPerGrid = N / threadsPerBlock;

    dim3 blocks_case2(blocksPerGrid);
    dim3 threads_case2(threadsPerBlock);

    cudaMemset(d_C, 0, size);

    addArrays << <blocks_case2, threads_case2 >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    print_results(h_C, h_A, h_B);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}*/








//Bonus Question

/*#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048
#define THREADS_PER_BLOCK 256

__global__ void addArrays(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { C[i] = A[i] + B[i]; }
}

__global__ void squareArray(int* C, int* D, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { D[i] = C[i] * C[i]; }
}

__global__ void sumReduction(int* input, int* output_sum, int n) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int thread_sum = 0;
    for (int i = global_idx; i < n; i += gridDim.x * blockDim.x) {
        thread_sum += input[i];
    }
    s_data[tid] = thread_sum;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output_sum, s_data[0]);
    }
}


int main() {
    int size = N * sizeof(int);

    int* h_A, * h_B, * h_D;
    int* d_A, * d_B, * d_C, * d_D;

    int h_sum = 0;
    int* d_sum;

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_D = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_D, size);
    cudaMalloc(&d_sum, sizeof(int));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // MODIFICATION: Initialize d_sum using cudaMemcpy instead of cudaMemset
    cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int threads = THREADS_PER_BLOCK;

    addArrays << <blocks, threads >> > (d_A, d_B, d_C, N);
    squareArray << <blocks, threads >> > (d_C, d_D, N);

    size_t shared_mem_size = threads * sizeof(int);
    sumReduction << <blocks, threads, shared_mem_size >> > (d_D, d_sum, N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    long long expected_sum = 0;
    for (int i = 0; i < N; i++) {
        long long c_val = h_A[i] + h_B[i];
        expected_sum += c_val * c_val;
    }

    printf("GPU Reduction Sum: %d\n", h_sum);
    printf("CPU Expected Sum:  %lld\n", expected_sum);

    if (h_sum == expected_sum) {
        printf("Success! The sum is correct. ✅\n");
    }
    else {
        printf("Error! The sum is incorrect. ❌\n");
    }

    free(h_A);
    free(h_B);
    free(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_sum);

    return 0;
}*/

