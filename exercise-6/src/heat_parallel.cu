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
#define N_ITEMS (M + 2) * (N + 2)
#define MAX_THREADS 1024

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
#define d_T(x, y) d_temp[(y) * (N + 2) + (x)]
#define d_THERMAL_DIFFUSIVITY(x, y) d_thermal_diffusivity[(y) * (N + 2) + (x)]

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

void time_step();
void boundary_condition();
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

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // TODO 6: Launch the cooperative time_step-kernel.

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // TODO 7: Copy data from device to host.

            domain_save(iteration);
        }
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    exit(EXIT_SUCCESS);
}

// TODO 4: Make time_step() a cooperative CUDA kernel
//         where one thread is responsible for one grid point.
void time_step()
{
    real_t c, t, b, l, r, K, A, D, new_value;

    for (int_t y = 1; y <= M; y++)
    {
        int res = y % 2;
        for (int_t x = 1 + res; x <= N; x += 2)
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            A = -K * dt;
            D = 1.0f + 4.0f * K * dt;

            new_value = (c - A * (t + b + l + r)) / D;

            T(x, y) = new_value;
        }
    }

    for (int_t y = 1; y <= M; y++)
    {
        int res = (y + 1) % 2;
        for (int_t x = 1 + res; x <= N; x += 2)
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            A = -K * dt;
            D = 1.0f + 4.0f * K * dt;

            new_value = (c - A * (t + b + l + r)) / D;

            T(x, y) = new_value;
        }
    }
}

// TODO 5: Make boundary_condition() a device function and
//         call it from the time_step-kernel.
//         Chose appropriate threads to set the boundary values.
void boundary_condition()
{
    for (int_t x = 1; x <= N; x++)
    {
        T(x, 0) = T(x, 2);
        T(x, M + 1) = T(x, M - 1);
    }

    for (int_t y = 1; y <= M; y++)
    {
        T(0, y) = T(2, y);
        T(N + 1, y) = T(N - 1, y);
    }
}

void domain_init(void)
{
    h_temp = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));
    h_thermal_diffusivity = (real_t *)malloc((M + 2) * (N + 2) * sizeof(real_t));

    // Allocate device memory.
    gpu_error = cudaMalloc(&d_temp, (M + 2) * (N + 2) * sizeof(real_t));
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

            T(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }

    // Copy data from host to device.
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

    // Free device memory.
    cudaFree(&d_temp);
    cudaFree(&d_thermal_diffusivity);

}
