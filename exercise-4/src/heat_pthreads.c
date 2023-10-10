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

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t* boundary_buffer[THREAD_COUNT * 2];

typedef struct {
    int_t id;
    int_t offset;
    int_t local_N;
    real_t *temp[2];
    real_t *thermal_diffusivity;
    real_t dt;
} thread_info;

pthread_barrier_t barrier;
pthread_mutex_t file_lock;

#define T(x, y) info->temp[0][(y) * (info->local_N + 2) + (x)]
#define T_next(x, y) info->temp[1][((y) * (info->local_N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x, y) info->thermal_diffusivity[(y) * (info->local_N + 2) + (x)]

void time_step(thread_info*);
void boundary_condition(thread_info*);
void border_exchange(thread_info*);
void domain_init(thread_info*);
void domain_save(int_t, thread_info*);
void domain_finalize(thread_info*);
void app(thread_info*);

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
    pthread_mutex_init(&file_lock, NULL);

    pthread_t threads[THREAD_COUNT];
    for (int_t i = 0; i < THREAD_COUNT; i++) {
        thread_info* info = malloc(sizeof(thread_info));
        *info = (thread_info) {
            i,
            N / THREAD_COUNT * i,
            N / THREAD_COUNT,
            {NULL, NULL},
            NULL,
            0.0,
        };
        //printf("%d: %x\n", i, *info);
        pthread_create(&threads[i], NULL, &app, info);
    }

    for (int_t i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Done!\n");

    exit(EXIT_SUCCESS);
}

void app(thread_info* info) {
    //printf("%d ready\n", info->local_N);
    domain_init(info);
    border_exchange(info);

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);

    for (int_t iteration = 0; iteration <= max_iteration; iteration++)
    {
        border_exchange(info);
        boundary_condition(info);

        time_step(info);

        if (iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t)iteration / (real_t)max_iteration);

            domain_save(iteration, info);
        }

        swap(&info->temp[0], &info->temp[1]);
    }

    gettimeofday(&t_end, NULL);
    printf("Total elapsed time: %lf seconds\n",
           WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize(info);
}

void border_exchange(thread_info* info) {
    //printf("%d: %d\n", info->id, info->id - 1 + (info->id % 2 * 2));
    boundary_buffer[info->id * 2] = &T(1, 1);
    boundary_buffer[info->id * 2 + 1] = &T(info->local_N, 1);
    
    pthread_barrier_wait(&barrier);

    if (info->id > 0) {
        // Copy from left
        memcpy(&T(0,1), boundary_buffer[info->id * 2 - 1], info->local_N * sizeof(real_t));
    }

    if (info->id < THREAD_COUNT - 1) {
        // Copy from right
        memcpy(&T(info->local_N+1,1), boundary_buffer[info->id * 2 + 2], info->local_N * sizeof(real_t));
    }
}

void time_step(thread_info* info)
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

void boundary_condition(thread_info* info)
{
    for (int_t x = 1; x <= info->local_N; x++)
    {
        T(x, 0) = T(x, 2);
        T(x, M + 1) = T(x, M - 1);
    }

    for (int_t y = 1; y <= M; y++)
    {
        T(0, y) = T(2, y);
        T(info->local_N + 1, y) = T(info->local_N - 1, y);
    }
}

void domain_init(thread_info* info)
{
    info->temp[0] = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));
    info->temp[1] = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));
    info->thermal_diffusivity = malloc((M + 2) * (info->local_N + 2) * sizeof(real_t));

    info->dt = 0.1;

    for (int_t y = 1; y <= M; y++)
    {
        for (int_t x = 1; x <= info->local_N; x++)
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x - info->offset + y) / 20.0)) / 605.0;

            T(x, y) = temperature;
            T_next(x, y) = temperature;
            THERMAL_DIFFUSIVITY(x, y) = diffusivity;
        }
    }
}

void domain_save(int_t iteration, thread_info* info)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    pthread_mutex_lock(&file_lock);

    FILE *out = fopen(filename, "a");
    fseek(out, (info->local_N+2) * (M+2) * info->id, SEEK_SET);
    if (!out)
    {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        pthread_mutex_unlock(&file_lock); // Just in case
        exit(1);
    }

    fwrite(info->temp[0], sizeof(real_t), (info->local_N+2) * (M+2), out);
    fclose(out);

    pthread_mutex_unlock(&file_lock);
}

void domain_finalize(thread_info* info)
{
    free(info->temp[0]);
    free(info->temp[1]);
    free(info->thermal_diffusivity);
    free(info);
}
