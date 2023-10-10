#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)
#define THREAD_COUNT 8
#define RANK_FIRST (info->id == 0)
#define RANK_LAST (info->id == THREAD_COUNT - 1)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency,
    local_N;

real_t
    *temp[2] = {NULL, NULL},
    *thermal_diffusivity,
    dt = 0.1;

typedef struct
{
    int_t id;
} thread_info;

pthread_barrier_t barrier;
struct timeval t_start, t_end;

#define T(x, y) temp[0][(y) * (N + 2) + (x)]
#define T_next(x, y) temp[1][((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) thermal_diffusivity[(y) * (N + 2) + (x)]

void time_step(thread_info *);
void boundary_condition(thread_info *);
void border_exchange(void);
void domain_init(void);
void domain_save(int_t);
void domain_finalize(void);
void app(thread_info *);

void swap(real_t **m1, real_t **m2)
{
    real_t *tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int main(int argc, char **argv)
{
    pthread_barrier_init(&barrier, NULL, THREAD_COUNT);

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
    local_N = N / THREAD_COUNT;
    domain_init();

    gettimeofday(&t_start, NULL);

    pthread_t threads[THREAD_COUNT];
    for (int_t i = 0; i < THREAD_COUNT; i++)
    {
        thread_info *info = malloc(sizeof(thread_info));
        *info = (thread_info){
            i,
        };
        // printf("%d: %x\n", i, *info);
        pthread_create(&threads[i], NULL, &app, info);
    }

    for (int_t i = 0; i < THREAD_COUNT; i++)
    {
        pthread_join(threads[i], NULL);
    }
    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();
    exit(EXIT_SUCCESS);
}

void app(thread_info *info)
{
    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        if (RANK_FIRST)
        {
            boundary_condition(info);
        }
        pthread_barrier_wait(&barrier);
        time_step(info);

        pthread_barrier_wait(&barrier);
        if (iteration % snapshot_frequency == 0 && RANK_FIRST)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            domain_save(iteration);
        }

        if (RANK_FIRST)
        {
            swap(&temp[0], &temp[1]);
        }

    }
}

void time_step(thread_info *info)
{
    real_t c, t, b, l, r, K, new_value;

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = local_N * info->id; x <= local_N * (info->id + 1); x++)
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

void boundary_condition(thread_info *info)
{
    for ( int_t x = 1; x <= N; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, M+1) = T(x, M-1);
    }

    for ( int_t y = 1; y <= M; y++ )
    {
        T(0, y) = T(2, y);
        T(N+1, y) = T(N-1, y);
    }
}

void domain_init(void)
{
    temp[0] = malloc((M + 2) * (N + 2) * sizeof(real_t));
    temp[1] = malloc((M + 2) * (N + 2) * sizeof(real_t));
    thermal_diffusivity = malloc((M + 2) * (N + 2) * sizeof(real_t));

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= N; x++)
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
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

    fwrite(temp[0], sizeof(real_t), (N + 2) * (M + 2), out);
    fclose(out);
}

void domain_finalize(void)
{
    free(temp[0]);
    free(temp[1]);
    free(thermal_diffusivity);
    pthread_barrier_destroy(&barrier);
}
