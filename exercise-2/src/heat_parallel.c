#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    M,
    max_iteration,
    snapshot_frequency,
    size,
    rank,
    origin_N,
    origin_M,
    size_N,
    size_M;

int_t *local_sizes_N;
int_t *local_sizes_M;

real_t
    *temp[2] = {NULL, NULL},
    *thermal_diffusivity,
    dx,
    dt;

#define T(i, j) temp[0][(i) * (size_M + 2) + (j)]
#define T_next(i, j) temp[1][((i) * (size_M + 2) + (j))]
#define THERMAL_DIFFUSIVITY(i, j) thermal_diffusivity[(i) * (size_M + 2) + (j)]

void time_step(void);
void boundary_condition(void);
void border_exchange(void);
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
    // TODO 1: Initialize MPI
    MPI_Init(&argc, &argv);
    const int root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // TODO 2: Parse arguments in the rank 0 processes
    // and broadcast to other processes
    if (rank == root)
    {
        OPTIONS *options = parse_args(argc, argv);
        if (!options)
        {
            fprintf(stderr, "Argument parsing failed\n");
            exit(1);
        }

        N = options->N;
        M = options->M;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    MPI_Bcast(&N, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT, root, MPI_COMM_WORLD);


    local_sizes_N = malloc(size * sizeof(int_t));
    for (int_t r = 0; r < size; r++)
        local_sizes_N[r] = (int_t)(N / size) + ((r < (N % size)) ? 1 : 0);

    local_sizes_M = malloc(size * sizeof(int_t));
    for (int_t r = 0; r < size; r++)
        local_sizes_M[r] = (int_t)(M / size) + ((r < (M % size)) ? 1 : 0);

    // TODO 3: Allocate space for each process' sub-grids
    // and initialize data for the sub-grids
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // TODO 7: Communicate border values
        border_exchange();

        // TODO 5: Boundary conditions
        boundary_condition();

        // TODO 4: Time step calculations
        time_step();

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            // TODO 6 MPI I/O
            domain_save(iteration);
        }

        swap(&temp[0], &temp[1]);
    }
    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();


    // TODO 1: Finalize MPI
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

void time_step(void)

{
    // TODO 4: Time step calculations
    real_t c, t, b, l, r, K, new_value;

    for (int_t x = 1; x <= size_N; x++)
    {
        for (int_t y = 1; y <= size_M; y++)
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}

void boundary_condition(void)
{
    // TODO 5: Boundary conditions
    if (rank == 0 || rank == size - 1)
    {
        for (int_t x = 1; x <= size_N; x++)
        {
            T(x, 0) = T(x, 2);
            T(x, size_M + 1) = T(x, size_M - 1);
        }

        for (int_t y = 1; y <= size_M; y++)
        {
            T(0, y) = T(2, y);
            T(size_N + 1, y) = T(size_N - 1, y);
        }
    }
}

void border_exchange(void)
{
    // TODO 7: Communicate border values

    int_t next = (rank + size + 1) % size;
    int_t previous = (rank + size - 1) % size;
    int_t first = temp[0][1];
    first = 1;
    int_t new_first = 0;
    int_t last = temp[0][(size_N + 2) * (size_M + 2) - 2];
    last = 2;
    int_t new_last = 0;

    if (previous < rank)
    {
        // Sendrecv first
        MPI_Sendrecv(
            &first, 1, MPI_INT, previous, rank,
            &new_first, 1, MPI_INT, previous,
            previous, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (next > rank)
    {
        // Sendrecv last
        MPI_Sendrecv(
            &last, 1, MPI_INT, next, rank,
            &new_last, 1, MPI_INT, next,
            next, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    temp[0][0] = new_first;
    temp[0][(size_N + 2) * (size_M + 2) - 1] = new_last;

}

void domain_init(void)
{
    // TODO 3: Allocate space for each process' sub-grids
    // and initialize data for the sub-grids
    size_N = local_sizes_N[rank];
    size_M = local_sizes_M[rank];
    origin_N = 0;
    origin_M = 0;

    for (int_t i = 0; i < rank; i++)
    {
        origin_N += local_sizes_N[i];
        origin_M += local_sizes_M[i];
    }

    real_t
        temperature,
        diffusivity;

    temp[0] = malloc((size_N + 2) * (size_M + 2) * sizeof(real_t));
    temp[1] = malloc((size_N + 2) * (size_M + 2) * sizeof(real_t));
    thermal_diffusivity = malloc((size_N + 2) * (size_M + 2) * sizeof(real_t));

    dt = 0.1;
    dx = 0.1;

    for (int_t x = origin_N + 1; x <= origin_N + size_N; x++)
    {
        // printf("Rank %dx: %d\n", rank, x);
        for (int_t y = origin_M + 1; y <= origin_M + size_M; y++)
        {
            // printf("Rank %dy: %d\n", rank, y);
            temperature = 30 + 30 * sin((x + y) / 20.0);
            diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;
            T(x - origin_N, y - origin_M) = temperature;
            T_next(x - origin_N, y - origin_M) = temperature;

            THERMAL_DIFFUSIVITY(x - origin_N, y - origin_M) = diffusivity;
        }
    }
}

void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    MPI_File out;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &out);
    if (!out)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    int_t offset = 0;
    for (int_t i = 0; i < rank; i++)
    {
        offset += (size_N + 2) * (size_M + 2);
    }
    MPI_File_write_at_all(out, offset, temp[0], (size_N + 2) * (size_M + 2), MPI_INT, MPI_STATUS_IGNORE);
    // fwrite(temp[0], sizeof(real_t), (N + 2) * (M + 2), out);
    printf("Writing Rank %d iteration %d, %s\n", rank, offset, filename);
    MPI_File_close(&out);
}

void domain_finalize(void)
{
    free(temp[0]);
    free(temp[1]);
    free(thermal_diffusivity);
}
