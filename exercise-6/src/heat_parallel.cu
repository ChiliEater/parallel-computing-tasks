#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include "../inc/argument_utils.h"
using namespace cooperative_groups;
namespace cg = cooperative_groups;

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define UPLOAD cudaMemcpyHostToDevice
#define DOWNLOAD cudaMemcpyDeviceToHost
#define GPU2GPU cudaMemcpyDeviceToDevice
#define THREAD_X (blockDim.x * blockIdx.x + threadIdx.x)
#define THREAD_Y (blockDim.y * blockIdx.y + threadIdx.y)
#define OUT_OF_BOUNDS (x > N || y > M || x == 0 || y == 0)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    dt,
    *h_temp,
    *h_thermal_diffusivity,
    // Declare device side pointers to store host-side data.
    *d_temp,
    *d_thermal_diffusivity;

cudaError_t gpu_error;

#define T(x, y) h_temp[(y) * (N + 2) + (x)]
#define THERMAL_DIFFUSIVITY(x, y) h_thermal_diffusivity[(y) * (N + 2) + (x)]
#define d_T(x, y) temp[(y) * (N + 2) + (x)]
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

__global__ void time_step(real_t *, real_t *, int_t, int_t, real_t);
__device__ void boundary_condition(real_t *, int_t, int_t);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);

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
    dim3 gridDims = {
        (unsigned int)ceil((double)N / (double)32.0),
        (unsigned int)ceil((double)M / (double)32.0),
        1};

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // Launch the cooperative time_step-kernel.
        void *gpu_args[] = {
            (void *)&d_temp,
            (void *)&d_thermal_diffusivity,
            (void *)&N,
            (void *)&M,
            (void *)&dt};
        cudaLaunchCooperativeKernel((void *)time_step, gridDims, threadBlockDims, gpu_args);

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // Download data from GPU to host.
            gpu_error = cudaMemcpy(h_temp, d_temp, (M + 2) * (N + 2) * sizeof(real_t), DOWNLOAD);
            cudaErrorCheck(gpu_error);
            domain_save(iteration);
        }
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    exit(EXIT_SUCCESS);
}

// Make time_step() a cooperative CUDA kernel where one thread is responsible for one grid point.
__global__ void time_step(
    real_t *temp,
    real_t *thermal_diffusivity,
    int_t N,
    int_t M,
    real_t dt)
{
    cg::grid_group grid = cg::this_grid();
    real_t c, t, b, l, r, K, A, D, new_value;
    // Coordinates are off by one
    int x = THREAD_X + 1;
    int y = THREAD_Y + 1;

    boundary_condition(temp, N, M);

    // Is this pixel responsible for red or black?
    bool is_red = (x % 2) != (y % 2);

    // Red part of the algorithm
    if (is_red && !OUT_OF_BOUNDS)
    {
        c = d_T(x, y);

        t = d_T(x - 1, y);
        b = d_T(x + 1, y);
        l = d_T(x, y - 1);
        r = d_T(x, y + 1);
        K = d_THERMAL_DIFFUSIVITY(x, y);

        A = -K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        d_T(x, y) = new_value;
    }

    // We have to wait for red to finish
    grid.sync();

    // Black part of the algorithm
    if (!is_red && !OUT_OF_BOUNDS)
    {
        c = d_T(x, y);

        t = d_T(x - 1, y);
        b = d_T(x + 1, y);
        l = d_T(x, y - 1);
        r = d_T(x, y + 1);
        K = d_THERMAL_DIFFUSIVITY(x, y);

        A = -K * dt;
        D = 1.0f + 4.0f * K * dt;

        new_value = (c - A * (t + b + l + r)) / D;

        d_T(x, y) = new_value;
    }
}

// Make boundary_condition() a device function and
//         call it from the time_step-kernel.
//         Chose appropriate threads to set the boundary values.
__device__ void boundary_condition(
    real_t *temp,
    int_t N,
    int_t M)
{
    // Coordinates are off by one
    int x = THREAD_X + 1;
    int y = THREAD_Y + 1;

    // This is just like the PS5 suggested solution
    if (x == 1)
        d_T(x - 1, y) = d_T(x + 1, y);
    if (y == 1)
        d_T(x, y - 1) = d_T(x, y + 1);
    if (x == N)
        d_T(x + 1, y) = d_T(x - 1, y);
    if (y == M)
        d_T(x, y + 1) = d_T(x, y - 1);
}

void domain_init(void)
{
    // Allocate host memory
    h_temp = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_thermal_diffusivity = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));

    // Allocate GPU memory
    gpu_error = cudaMalloc(&d_temp, (M + 2) * (N + 2) * sizeof(real_t));
    cudaErrorCheck(gpu_error);
    gpu_error = cudaMalloc(&d_thermal_diffusivity, (M + 2) * (N + 2) * sizeof(real_t));
    cudaErrorCheck(gpu_error);

    // Starting data
    dt = 0.1;

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= N; x++)
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            T(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }

    // Upload data from host to GPU
    gpu_error = cudaMemcpy(d_temp, h_temp, (M + 2) * (N + 2) * sizeof(real_t), UPLOAD);
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
    fwrite(h_temp, sizeof(real_t), (N + 2) * (M + 2), out);
    fclose(out);
}

void domain_finalize(void)
{
    free(h_temp);
    free(h_thermal_diffusivity);

    // Free GPU memory.
    cudaFree(&d_temp);
    cudaFree(&d_thermal_diffusivity);
}
