/*
    Aron de Ruijter 12868655, Lars Janssen 12882712
    simulate.c

    This program simulates a wave equation in parallel using OpenMP.
    This is done using pragmas, these can be found in the simulate function.
*/

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"
#include "omp.h"

/*
 * This function swaps the pointers of the arrays.
 */
void swap(double **old_array, double **current_array, double **next_array)
{
    double *temp = *old_array;
    *old_array = *current_array;
    *current_array = *next_array;
    *next_array = temp;
}

/*
 * Executes the entire simulation.
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    omp_set_num_threads(num_threads);

    for (int t = 0; t < t_max; t++)
    {
        #pragma omp parallel for
        for (int i = 1; i < i_max; i++)
        {
            next_array[i] = 2 * current_array[i] - old_array[i] + 0.15 *
                            (current_array[i - 1] - (2 * current_array[i]
                            - current_array[i + 1]));
        }

        swap(&old_array, &current_array, &next_array);
    }

    return current_array;
}
