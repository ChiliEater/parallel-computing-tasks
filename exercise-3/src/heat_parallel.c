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

MPI_Comm communicator;

MPI_Datatype row, column;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency,
    local_N,
    local_M,
    x_offset,
    y_offset;
int
    size,
    rank;

int location[2];
int dimensions[] = {0, 0};

real_t
    *temp[2] = {NULL, NULL},
    *thermal_diffusivity,
    dt;

#define T(x, y) temp[0][(y) * (local_N + 2) + (x)]
#define T_next(x, y) temp[1][((y) * (local_N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) thermal_diffusivity[(y) * (local_N + 2) + (x)]
#define DIMENSIONS 2
#define MPI_RANK_FIRST (location[0] == 0 && location[1] == 0)
#define MPI_RANK_LAST ((location[0] == dimensions[0]) && (location[1] == dimensions[1]))
#define MPI_X_AXIS 0
#define MPI_Y_AXIS 1

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

// Open a log for the current rank
FILE *open_log()
{
    char name[] = "rank00.log";
    sprintf(name, "rank%d.log", rank);
    return fopen(name, "a");
}

// Close a log for the current rank
void close_log(FILE *file)
{
    fclose(file);
}

// CLear a log for the current rank via deletion
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
    // Valgrind tells me this call is creating memory leaks. No clue what that's about.
    MPI_Init(&argc, &argv);

    // Determine dimensions
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Dims_create(
        size,
        DIMENSIONS,
        dimensions);

    // Create communicator
    int periods[2] = {0, 0};

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

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        border_exchange();
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

    // Simply adjusting M and N to the local variants appears to do the trick
    for (int_t y = 1; y <= local_M; y++)
    {
        for (int_t x = 1; x <= local_N; x++)
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

void derive_types(void) {
    MPI_Type_vector(
        local_M,
        1,
        local_N+2,
        MPI_DOUBLE,
        &column
    );
    
    MPI_Type_contiguous(
        local_N,
        MPI_DOUBLE,
        &row
    );

    MPI_Type_commit(&column);
    MPI_Type_commit(&row);
}

void border_exchange(void) {
    int left;
    int right;
    int up;
    int down;

    MPI_Cart_shift(
        communicator,
        MPI_X_AXIS,
        1,
        &left,
        &right
    );
    MPI_Cart_shift(
        communicator,
        MPI_Y_AXIS,
        1,
        &up,
        &down
    );

    MPI_Sendrecv(&T(1, 1),
                1,
                column,
                left,
                0,
                &T(local_N+1, 1),
                1,
                column,
                right,
                0,
                communicator,
                MPI_STATUS_IGNORE);

    MPI_Sendrecv(&T(local_N, 1),
                1,
                column,
                right,
                0,
                &T(0, 1),
                1,
                column,
                left,
                0,
                communicator,
                MPI_STATUS_IGNORE);

    MPI_Sendrecv(&T(1, 1),
                1,
                row,
                up,
                0,
                &T(1, local_M+1),
                1,
                row,
                down,
                0,
                communicator,
                MPI_STATUS_IGNORE);

    MPI_Sendrecv(&T(1, local_M),
                1,
                row,
                down,
                0,
                &T(1, 0),
                1,
                row,
                up,
                0,
                communicator,
                MPI_STATUS_IGNORE);
}

void boundary_condition(void)
{
    // Regular boundary conditions
    for (int_t x = 1; x <= local_N; x++)
    {
        T(x, 0) = T(x, 2);
        T(x, local_M + 1) = T(x, local_M - 1);
    }

    for (int_t y = 1; y <= local_M; y++)
    {
        T(0, y) = T(2, y);
        T(local_N + 1, y) = T(local_N - 1, y);
    }

    // Special cases for the top left and bottom right ranks
    if (MPI_RANK_FIRST) {
        for ( int_t y = 1; y <= local_M; y++ )
        {
            T(0, y) = T(2, y);
        }
    }
    if (MPI_RANK_LAST) {
        for ( int_t y = 1; y <= local_M; y++ )
        {
            T(local_N+1, y) = T(local_N-1, y);
        }

    }
}

void domain_init(void)
{
    // Calculate offsets and sizes
    local_N = N / dimensions[0];
    local_M = M / dimensions[1];
    x_offset = local_N * location[0];
    y_offset = local_M * location[1];

    // Allocate enough memory
    // TODO: Valgrind tells me something isn't right here. False positive?
    temp[0] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    temp[1] = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));
    thermal_diffusivity = malloc((local_M + 2) * (local_N + 2) * sizeof(real_t));

    dt = 0.1;

    // Fill the the buffers with samples
    for (int_t y = 1; y <= local_M; y++)
    {
        for (int_t x = 1; x <= local_N; x++)
        {
            real_t temperature = 30 + 30 * sin((x_offset + x + y_offset + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x_offset + x + y_offset + y) / 20.0)) / 605.0;
            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
}

void domain_save(int_t iteration)
{
    // Determine file to write
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    // Get file descriptor
    MPI_File out;
    MPI_File_open (
        communicator,
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &out
    );

    // Ensure we actually opened the file
    if (!out)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    // Determine buffer sizes
    int_t load_offset = local_M+2;
    int_t load_size = local_N*(local_M+2);

    if ( MPI_RANK_FIRST )
    {
        load_size += local_M+2;
        load_offset = 0;
    }

    if ( MPI_RANK_LAST )
    {
        load_size += local_M+2;
    }

    // Write data
    MPI_File_write_ordered (
        out,
        temp[0] + load_offset,
        load_size,
        MPI_DOUBLE,
        MPI_STATUS_IGNORE
    );

    // Close file
    MPI_File_close ( &out );
}

void domain_finalize(void)
{
    free(temp[0]);
    free(temp[1]);
    free(thermal_diffusivity);
}
