#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define UPLOAD cudaMemcpyHostToDevice
#define DOWNLOAD cudaMemcpyDeviceToHost
#define GPU2GPU cudaMemcpyDeviceToDevice
#define N_ITEMS (M + 2) * (N + 2)
#define MAX_THREADS 1024

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

cudaError_t gpu_error;

real_t
    *h_temp[2] = {NULL, NULL},
    *h_thermal_diffusivity,
    *d_temp,
    *d_temp_next,
    *d_thermal_diffusivity,
    dt;

#define T(x, y) h_temp[(y) * (N + 2) + (x)]
#define T_next(x, y) h_temp_next[((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) h_thermal_diffusivity[(y) * (N + 2) + (x)]

#define d_T(x, y) temp[(y) * (N + 2) + (x)]
#define d_T_next(x, y) temp_next[((y) * (N + 2) + (x))]
#define d_THERMAL_DIFFUSIVITY(x, y) thermal_diffusivity[(y) * (N + 2) + (x)]

#define cudaErrorCheck(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void time_step(real_t *, real_t *, real_t *, int_t, int_t);
__device__ void boundary_condition(real_t *, real_t *, int, int, int, int);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);

void swap(real_t **m1, real_t **m2)
{
    real_t *tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int main(int argc, char **argv)
{
    OPTIONS *options = parse_args(argc, argv);
    if (!options)
    {
        fprintf(stderr, "Argument parsing failed\n");
        exit(1);
    }

    M = options->M;
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    // 32 x 32 = 1024 which is the max per block
    dim3 threadBlockDims = {32, 32, 1};
    // Split the domain into 32-large chunks and round up to capture everything
    dim3 gridDims = {ceil(N / 32), ceil(M / 32), 1};

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // Launch the time_step-kernel.
        time_step<<<gridDims, threadBlockDims>>>(d_temp, d_temp_next, d_thermal_diffusivity, N, M);
        cudaDeviceSynchronize(); // Is this even necessary???
        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // Copy data from device to host.
            gpu_error = cudaMemcpy(h_temp[0], d_temp, (M + 2) * (N + 2) * sizeof(real_t), DOWNLOAD);
            cudaErrorCheck(gpu_error);
            gpu_error = cudaMemcpy(h_temp[1], d_temp_next, (M + 2) * (N + 2) * sizeof(real_t), DOWNLOAD);
            cudaErrorCheck(gpu_error);

            domain_save(iteration);
        }
        //  Swap device pointers.
        swap(&d_temp, &d_temp_next);
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    exit(EXIT_SUCCESS);
}

// Make time_step() a CUDA kernel
//         where one thread is responsible for one grid point.
__global__ void time_step(
    real_t *temp,
    real_t *temp_next,
    real_t *thermal_diffusivity,
    int_t N,
    int_t M)
{
    // From slides, should work fine, right?
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // This *should* stop threads that are outside our domain
    // i.e. counteracting the rounding we do in main
    if (x > (N + 1) || y > (M + 1))
        return;

    boundary_condition(temp, temp_next, x, y, N, M);
    
    // Should this be here?
    //if (x == 0 || y == 0)
    //    return;

    real_t c, t, b, l, r, K, new_value;
    // Probably should define this as a constant
    real_t dt = 0.1;

    c = d_T(x, y);
    t = d_T(x - 1, y);
    b = d_T(x + 1, y);
    l = d_T(x, y - 1);
    r = d_T(x, y + 1);
    K = d_THERMAL_DIFFUSIVITY(x, y);

    new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

    d_T_next(x, y) = new_value;
}

// Make boundary_condition() a device function and
//         call it from the time_step-kernel.
//         Chose appropriate threads to set the boundary values.
__device__ void boundary_condition(real_t *temp, real_t *temp_next, int x, int y, int N, int M)
{
    // Top row
    if (y == 0)
    {
        d_T(x, 0) = d_T(x, 2);
    }
    // Bottom row
    if (y == M + 1)
    {
        d_T(x, M + 1) = d_T(x, M - 1);
    }
    // Leftmost column
    if (x == 0)
    {
        d_T(0, y) = d_T(2, y);
    }
    // Rightmost column
    if (x == N + 1)
    {
        d_T(N + 1, y) = d_T(N - 1, y);
    }
}

void domain_init(void)
{
    h_temp[0] = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_temp[1] = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_thermal_diffusivity = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));

    // Allocate device memory.
    gpu_error = cudaMalloc(&d_temp, (M + 2) * (N + 2) * sizeof(real_t));
    cudaErrorCheck(gpu_error);
    gpu_error = cudaMalloc(&d_temp_next, (M + 2) * (N + 2) * sizeof(real_t));
    cudaErrorCheck(gpu_error);
    gpu_error = cudaMalloc(&d_thermal_diffusivity, (M + 2) * (N + 2) * sizeof(real_t));
    cudaErrorCheck(gpu_error);

    dt = 0.1;

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= N; x++)
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            h_temp[0][y * (N + 2) + x] = temperature;
            h_temp[1][y * (N + 2) + x] = temperature;
            h_thermal_diffusivity[y * (N + 2) + x] = diffusivity;
        }
    }
    // Copy data from host to device.
    gpu_error = cudaMemcpy(d_temp, h_temp[0], (M + 2) * (N + 2) * sizeof(real_t), UPLOAD);
    cudaErrorCheck(gpu_error);
    gpu_error = cudaMemcpy(d_temp_next, h_temp[1], (M + 2) * (N + 2) * sizeof(real_t), UPLOAD);
    cudaErrorCheck(gpu_error);
    gpu_error = cudaMemcpy(d_thermal_diffusivity, h_thermal_diffusivity, (M + 2) * (N + 2) * sizeof(real_t), UPLOAD);
    cudaErrorCheck(gpu_error);
}

void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    FILE *out = fopen(filename, "wb");
    if (!out)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    for (int_t iter = 1; iter <= N; iter++)
    {
        fwrite(h_temp[0] + (M + 2) * iter + 1, sizeof(real_t), N, out);
    }
    fclose(out);
}

void domain_finalize(void)
{
    free(h_temp[0]);
    free(h_temp[1]);
    free(h_thermal_diffusivity);

    // Free device memory.
    cudaFree(&d_temp);
    cudaFree(&d_temp_next);
    cudaFree(&d_thermal_diffusivity);
}
