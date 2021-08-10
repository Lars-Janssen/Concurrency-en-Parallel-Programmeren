/*
 * simulate.h
 *
 */

#pragma once

double *simulate(const int i_max, const int t_max, double *old_array,
        double *current_array, double *next_array);
void worker(int my_rank, int num_tasks, int i_max, int t_max);
void compute(int bufferSize, int processSize, int num_tasks, int my_rank,
             int t_max);