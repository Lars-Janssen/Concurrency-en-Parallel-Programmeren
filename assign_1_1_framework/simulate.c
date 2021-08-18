/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"
#include "thread.h"


/* Add any global variables you may need. */

double *global_old_array;
double *global_current_array;
double *global_next_array;
int global_t_max;

typedef struct
{
    int start, end;
    int id;
} t_data;

pthread_barrier_t barrier1;
pthread_barrier_t barrier2;

void swap(double **old_array, double **current_array, double **next_array)
{
    double *temp = *old_array;
    *old_array = *current_array;
    *current_array = *next_array;
    *next_array = temp;
}

void *thread(void *a)
{
    t_data *myData = (t_data*) a;
    for (int t = 0; t < global_t_max; t++)
    {
        for (int i = myData->start; i < myData->end - 1; i++)
        {
            global_next_array[i] = 2 * global_current_array[i]
            - global_old_array[i] + 0.15 * (global_current_array[i-1]
            - (2 * global_current_array[i] - global_current_array[i+1]));
        }
        pthread_barrier_wait(&barrier1);
        pthread_barrier_wait(&barrier2);
    }
    return a;
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    global_old_array = old_array;
    global_current_array = current_array;
    global_next_array = next_array;
    global_t_max = t_max;

    int i_end = (double)i_max / (double)num_threads - 1;

    pthread_t thread_ids[num_threads - 1];
    t_data myData[num_threads - 1];
    void *result;

    pthread_barrier_init(&barrier1, NULL, num_threads);
    pthread_barrier_init(&barrier2, NULL, num_threads);

    for (int i = 0; i < num_threads - 1; i++)
    {
        myData[i].id = i;
        myData[i].start = (i+1) * ((double)i_max / (double)num_threads) - 1;
        myData[i].end = (i+2) * ((double)i_max / (double)num_threads);
        pthread_create(&thread_ids[i], NULL, thread, (void*)&myData[i]);
    }

    for (int t = 0; t < t_max; t++)
    {
        for (int i = 1; i < i_end; i++)
        {
            global_next_array[i] = 2 * global_current_array[i]
            - global_old_array[i] + 0.15 * (global_current_array[i-1]
            - (2 * global_current_array[i] - global_current_array[i+1]));
        }

        pthread_barrier_wait(&barrier1);

        swap(&global_old_array, &global_current_array, &global_next_array);
        // for (int i = 0; i < i_max; i++)
        // {
        //     global_old_array[i] = global_current_array[i];
        //     global_current_array[i] = global_next_array[i];
        // }

        pthread_barrier_wait(&barrier2);
    }

    for (int i = 0; i < num_threads - 1; i++)
    {
        pthread_join(thread_ids[i], &result);
    }

    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);


    /* You should return a pointer to the array with the final results. */
    return global_current_array;
}
