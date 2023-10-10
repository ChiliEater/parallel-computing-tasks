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
#define THREAD_COUNT 4
#define RANK_FIRST (info->id == 0)
#define RANK_LAST (info->id == THREAD_COUNT - 1)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t *boundary_buffer[THREAD_COUNT * 2];

typedef struct
{
    int_t id;
    int_t offset;
    int_t local_N;
    real_t *temp[2];
    real_t *thermal_diffusivity;
    real_t dt;
} thread_info;

pthread_barrier_t barrier;
pthread_mutex_t file_lock;
pthread_barrier_t file_barrier;

#define T(x, y) info->temp[0][(y) * (info->local_N + 2) + (x)]
#define T_next(x, y) info->temp[1][((y) * (info->local_N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) info->thermal_diffusivity[(y) * (info->local_N + 2) + (x)]

void time_step(thread_info *);
void boundary_condition(thread_info *);
void border_exchange(thread_info *);
void domain_init(thread_info *);
void domain_save(int_t, thread_info *);
void domain_finalize(thread_info *);
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

    pthread_barrier_init(&barrier, NULL, THREAD_COUNT);
    pthread_barrier_init(&file_barrier, NULL, THREAD_COUNT);
    pthread_mutex_init(&file_lock, NULL);

    pthread_t threads[THREAD_COUNT];
    for (int_t i = 0; i < THREAD_COUNT; i++)
    {
        thread_info *info = malloc(sizeof(thread_info));
        *info = (thread_info){
            i,
            N / THREAD_COUNT * i,
            N / THREAD_COUNT,
            {NULL, NULL},
            NULL,
            0.0,
        };
        // printf("%d: %x\n", i, *info);
        pthread_create(&threads[i], NULL, &app, info);
    }

    for (int_t i = 0; i < THREAD_COUNT; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&file_lock);
    pthread_barrier_destroy(&barrier);
    pthread_barrier_destroy(&file_barrier);

    printf("Done!\n");

    exit(EXIT_SUCCESS);
}

void app(thread_info *info)
{
    domain_init(info);
    printf("%d ready\n", info->id);

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        border_exchange(info);
        boundary_condition(info);
        time_step(info);

        if (iteration % snapshot_frequency == 0)
        {
            if (RANK_LAST)
            {
                printf(
                    "Iteration %ld of %ld (%.2lf%% complete)\n",
                    iteration,
                    max_iteration,
                    100.0 * (real_t)iteration / (real_t)max_iteration);
            }

            domain_save(iteration, info);
        }

        swap(&info->temp[0], &info->temp[1]);
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize(info);
}

void border_exchange(thread_info *info)
{
    // printf("%d: %d\n", info->id, info->id - 1 + (info->id % 2 * 2));
    boundary_buffer[info->id * 2] = info->temp[0] + (M + 2);
    boundary_buffer[info->id * 2 + 1] = info->temp[0] + (M + 2) * info->local_N;

    pthread_barrier_wait(&barrier);

    if (!RANK_FIRST)
    {
        // Copy from up?
        //for (int i = 0; i < info->local_N + 2; i++) {
        //    T(i,0) = boundary_buffer[info->id * 2 - 1][i];
        //}
        memcpy(
            info->temp[0],
            boundary_buffer[info->id * 2 - 1],
            (info->local_N + 2) * sizeof(real_t));
    }

    if (!RANK_LAST)
    {
        // Copy from down?
        //for (int i = 0; i < info->local_N + 2; i++) {
        //    T(i+1,M + 2) = boundary_buffer[info->id * 2 + 2][i];
        //}
        memcpy(
            info->temp[0] + (M + 2) * (info->local_N + 1),
            boundary_buffer[info->id * 2 + 2],
            (info->local_N + 2) * sizeof(real_t));
    }
}

void time_step(thread_info *info)
{
    real_t c, t, b, l, r, K, new_value;

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= info->local_N; x++)
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * info->dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}

void boundary_condition(thread_info *info)
{
    for (int_t x = 1; x <= info->local_N; x++)
    {
        T(x, 0) = T(x, 2);
    }
    for (int_t x = 1; x <= info->local_N; x++)
    {
        T(x, M + 1) = T(x, M - 1);
    }

    if (RANK_FIRST)
    {
        for (int_t y = 1; y <= M; y++)
        {
            T(0, y) = T(2, y);
        }
    }
    if (RANK_LAST)
    {
        for (int_t y = 1; y <= M; y++)
        {
            T(info->local_N + 1, y) = T(info->local_N - 1, y);
        }
    }
}

void domain_init(thread_info *info)
{
    int_t remaining_N = N % THREAD_COUNT;
    printf("%d\n", info->local_N);
    if (remaining_N != 0)
    {
        if (info->id < remaining_N)
        {
            info->local_N++;
            info->offset += info->id;
        }
        else
        {
            info->offset += remaining_N;
        }
    }

    info->temp[0] = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));
    info->temp[1] = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));
    info->thermal_diffusivity = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));

    info->dt = 0.1;

    for (int_t x = 1; x <= info->local_N; x++)
    {
        for (int_t y = 1; y <= M; y++)
        {
            real_t temperature = 30 + 30 * sin((info->offset + x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + info->offset) + y) / 20.0)) / 605.0;

            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
}

void domain_save(int_t iteration, thread_info *info)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);
    pthread_mutex_lock(&file_lock);

    FILE *out = fopen(filename, "a");
    if (!out)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        pthread_mutex_unlock(&file_lock); // Just in case
        exit(1);
    }

    int_t load_offset = M + 2;
    int_t load_size = info->local_N * (M + 2);
    int_t seek_offset = load_size + M + 2 + load_size * (info->id - 1);

    if (RANK_FIRST)
    {
        load_size += M + 2;
        load_offset = 0;
        seek_offset = 0;
    }
    if (RANK_LAST)
    {
        load_size += M + 2;
    }

    fseek(out, seek_offset, SEEK_SET);
    fwrite(info->temp[0] + load_offset, sizeof(real_t), load_size, out);
    fclose(out);

    pthread_mutex_unlock(&file_lock);
    pthread_barrier_wait(&file_barrier);
}

void domain_finalize(thread_info *info)
{
    free(info->temp[0]);
    free(info->temp[1]);
    free(info->thermal_diffusivity);
    free(info);
}
