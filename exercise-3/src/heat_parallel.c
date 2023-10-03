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
    M,
    N,
    max_iteration,
    snapshot_frequency,
    size,
    rank,
    local_N,
    local_M,
    x_offset,
    y_offset;

int location[];
int dimensions[] = {0, 0};

real_t
    *temp[2] = {NULL, NULL},
    *thermal_diffusivity,
    dt;

#define T(x, y) temp[0][(y) * (local_M + 2) + (x)]
#define T_next(x, y) temp[1][((y) * (local_M + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) thermal_diffusivity[(y) * (N + 2) + (x)]
#define DIMENSIONS 2

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

FILE *open_log()
{
    char name[] = "rank00.log";
    sprintf(name, "rank%d.log", rank);
    return fopen(name, "a");
}

void close_log(FILE *file)
{
    fclose(file);
}

void clear_log()
{
    char name[] = "rank00.log";
    sprintf(name, "rank%d.log", rank);
    remove(name);
}

int main(int argc, char **argv)
{
    // TODO 1:
    // - Initialize and finalize MPI.
    // - Create a cartesian communicator.
    // - Parse arguments in the rank 0 processes
    //   and broadcast to other processes

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Determine dimensions
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(
        size,
        DIMENSIONS,
        dimensions);

    // Create communicator
    MPI_Comm communicator;
    int_t periods[2] = {0, 0};

    MPI_Cart_create(
        MPI_COMM_WORLD,
        DIMENSIONS,
        dimensions,
        periods,
        0,
        &communicator);

    // Initialize rank and position
    const int_t root = 0;
    MPI_Comm_rank(communicator, &rank);
    clear_log();

    MPI_Cart_coords(
        communicator,
        rank,
        DIMENSIONS,
        location);

    // Let root parse args
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

    // Broadcast results to evereyone else
    MPI_Bcast(&N, 1, MPI_INT, root, communicator);
    MPI_Bcast(&M, 1, MPI_INT, root, communicator);
    MPI_Bcast(&max_iteration, 1, MPI_INT, root, communicator);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT, root, communicator);

    domain_init();
    MPI_Finalize();
    exit(EXIT_SUCCESS);

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        // TODO 6: Implement border exchange.
        // Hint: Creating MPI datatypes for rows and columns might be useful.

        boundary_condition();

        time_step();

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            domain_save(iteration);
        }

        swap(&temp[0], &temp[1]);
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

void time_step(void)
{
    real_t c, t, b, l, r, K, new_value;

    // TODO 3: Update the area of iteration so that each
    // process only iterates over its own subgrid.

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= N; x++)
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
    // TODO 4: Change the application of boundary conditions
    // to match the cartesian topology.

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
    // TODO 2:
    // - Find the number of columns and rows in each process' subgrid.
    // - Allocate memory for each process' subgrid.
    // - Find each process' offset to calculate the correct initial values.
    // Hint: you can get useful information from the cartesian communicator.
    // Note: you are allowed to assume that the grid size is divisible by
    // the number of processes.
    local_N = N / dimensions[0];
    local_M = M / dimensions[1];
    x_offset = local_N * location[0];
    y_offset = local_M * location[1];

    temp[0] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    temp[1] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    thermal_diffusivity = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));

    //printf("nmxy: %d, %d: %d, %d\n", local_N, local_M, x_offset, y_offset);
    dt = 0.1;
    int loops = 0;
    for (int_t y = 1; y <= local_M; y++)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            real_t temperature = 30 + 30 * sin((x_offset + x + y_offset + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x_offset + x + y_offset + y) / 20.0)) / 605.0;
            //FILE *log = open_log();
            //fprintf(log, "(%d, %d) Accessing: %d, %d, index %d\n", location[0], location[1], x, y, (y) * (N + 2) + (x));
            //close_log(log);
            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
    //printf("loops: %d\n", loops);
    //printf("llt: %d, %d: %d\n", location[0], location[1], T(29, 69));
}

void domain_save(int_t iteration)
{
    // TODO 5: Use MPI I/O to save the state of the domain to file.
    // Hint: Creating MPI datatypes might be useful.

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
        fwrite(temp[0] + (M + 2) * iter + 1, sizeof(real_t), N, out);
    }
    fclose(out);
}

void domain_finalize(void)
{
    free(temp[0]);
    free(temp[1]);
    free(thermal_diffusivity);
}
