#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "../inc/argument_utils.h"
#include <mpi.h>

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

#define MPI_RANK_ROOT  ( rank == 0 )
#define MPI_RANK_FIRST ( MPI_RANK_ROOT )
#define MPI_RANK_LAST  ( rank == comm_size-1 )
#define MPI_RANK_EVEN  ( rank % 2 == 0 )
#define MPI_RANK_ODD   ( rank % 2 != 0 )

int
    rank,
    comm_size,
    local_N,
    remaining_N,
    local_x_offset;

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    M,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dx,
    dt;

#define T(i,j)                      temp[0][(i)*(M+2)+(j)]
#define T_next(i,j)                 temp[1][((i)*(M+2)+(j))]
#define THERMAL_DIFFUSIVITY(i,j)    thermal_diffusivity[(i)*(M+2)+(j)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );

void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int
main ( int argc, char **argv )
{
    // TODO 1: Initialize MPI
    MPI_Init( &argc, &argv );

    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // TODO 2: Parse arguments in the rank 0 processes
    // and broadcast to other processes
    if ( MPI_RANK_ROOT )
    {
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        N = options->N;
        M = options->M;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    MPI_Bcast ( &N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );

    // TODO 3: Allocate space for each process' sub-grids
    // and initialize data for the sub-grids
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 7: Communicate border values
        border_exchange();

        // TODO 5: Boundary conditions
        boundary_condition();

        // TODO 4: Time step calculations
        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            if ( MPI_RANK_ROOT )
            {
                printf (
                    "Iteration %ld of %ld (%.2lf%% complete)\n",
                    iteration,
                    max_iteration,
                    100.0 * (real_t) iteration / (real_t) max_iteration
                );
            }

            // TODO 6 MPI I/O
            domain_save ( iteration );
        }
        swap( &temp[0], &temp[1] );
    }
    gettimeofday ( &t_end, NULL );
    if ( MPI_RANK_ROOT )
    {
        printf ( "Total elapsed time: %lf seconds\n",
                WALLTIME(t_end) - WALLTIME(t_start)
                );
    }

    domain_finalize();

    // TODO 1: Finalize MPI
    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    // TODO 4: Time step calculations
    real_t c, t, b, l, r, K, new_value;

    for ( int_t x = 1; x <= local_N; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
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


void
boundary_condition ( void )
{
    // TODO 5: Boundary conditions
    for ( int_t x = 1; x <= local_N; x++ )
    {
        T(x, 0) = T(x, 2);
    }
    for ( int_t x = 1; x <= local_N; x++ )
    {
        T(x, M+1) = T(x, M-1);
    }

    if ( MPI_RANK_FIRST )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            T(0, y) = T(2, y);
        }
    }
    if ( MPI_RANK_LAST )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            T(local_N+1, y) = T(local_N-1, y);
        }
    }
}


void
border_exchange ( void )
{
    // TODO 7: Communicate border values
    int_t rank_prev = !MPI_RANK_FIRST ? rank-1 : MPI_PROC_NULL;
    int_t rank_next = !MPI_RANK_LAST ? rank+1 : MPI_PROC_NULL;

    MPI_Sendrecv(temp[0] + (M + 2),
                 M + 2,
                 MPI_DOUBLE,
                 rank_prev,
                 0,
                 temp[0] + (M + 2) * (local_N + 1),
                 M + 2,
                 MPI_DOUBLE,
                 rank_next,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(temp[0] + (M + 2) * local_N,
                 M + 2,
                 MPI_DOUBLE,
                 rank_next,
                 1,
                 temp[0],
                 M + 2,
                 MPI_DOUBLE,
                 rank_prev,
                 1,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}


void
domain_init ( void )
{
    local_N = N / comm_size;
    local_x_offset = rank * local_N;
    remaining_N = N % comm_size;
    if ( remaining_N != 0 )
    {
        if ( rank < remaining_N )
        {
            local_N ++;
            local_x_offset += rank;
        }
        else
        {
            local_x_offset += remaining_N;
        }
    }

    temp[0] = malloc ( (local_N+2)*(M+2) * sizeof(real_t) );
    temp[1] = malloc ( (local_N+2)*(M+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (local_N+2)*(M+2) * sizeof(real_t) );

    dt = 0.1;
    dx = 0.1;

    for ( int_t x = 1; x <= local_N; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            real_t temperature = 30+30*sin((local_x_offset+x+y)/20.0);
            real_t diffusivity = 0.05+(30+30*sin((N-(local_x_offset+x)+y)/20.0))/605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    // TODO 6: MPI I/O
    MPI_File out;
    MPI_File_open (
        MPI_COMM_WORLD,
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &out
    );

    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    int_t load_offset = M+2;
    int_t load_size = local_N*(M+2);

    if ( MPI_RANK_FIRST )
    {
        load_size += M+2;
        load_offset = 0;
    }
    if ( MPI_RANK_LAST )
    {
        load_size += M+2;
    }

    MPI_File_write_ordered (
        out,
        temp[0] + load_offset,
        load_size,
        MPI_DOUBLE,
        MPI_STATUS_IGNORE
    );

    MPI_File_close ( &out );
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
