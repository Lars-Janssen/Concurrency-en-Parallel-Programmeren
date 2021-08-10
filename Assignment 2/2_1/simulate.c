/*
    Aron de Ruijter 12868655 Lars Janssen 12882712

    This program uses MPI to calculate a wave equation in parallel.
    For more info on the design, please view the paper we turned in.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "simulate.h"

/*
    The local arrays and the send/recieve buffers.
*/
double *local_old, *local_current, *local_next;
double *send_buf, *recv_buf;
int workerSize;

/*
    This is a worker function. It gets its data from the master, calculates
    the equation, and returns the result bac to the master.
*/
void worker(int my_rank, int num_tasks, int i_max, int t_max)
{
    /*
        The arrays are 2 larger than the workerSize, because the first and
        last index are used as halo cells.
    */
    local_old = malloc((workerSize + 2) * sizeof(double));
    local_current = malloc((workerSize + 2) * sizeof(double));
    local_next = malloc((workerSize + 2) * sizeof(double));
    send_buf = malloc(workerSize * sizeof(double));
    recv_buf = malloc(workerSize * sizeof(double));

    /*
        This initializes the worker by sending it the part of old_array
        and current_array that it needs to work on.
    */
    MPI_Recv(&recv_buf[0], workerSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    memcpy(&local_old[1], &recv_buf[0], workerSize * sizeof(double));
    MPI_Recv(&recv_buf[0], workerSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    memcpy(&local_current[1], &recv_buf[0], workerSize * sizeof(double));

    /*
        This computes the results and sends the values on the edges to other processes.
    */
    compute(workerSize, workerSize, num_tasks, my_rank, t_max);

    /*
        This sends the resulting array back to the master.
    */
    memcpy(&send_buf[0], &local_current[1], workerSize * sizeof(double));
    MPI_Send(&send_buf[0], workerSize, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
}

/*
    This computes the result by calculating the wave for every time-step
    and sending the necessary values to other workers.
*/
void compute(int bufferSize, int processSize, int num_tasks, int my_rank,
             int t_max)
{
    int my_left = my_rank - 1;
    int my_right = my_rank + 1;

    for (int t = 0; t < t_max; t++)
    {
        send_buf[0] = local_current[1];
        send_buf[1] = local_current[processSize];

        /*
            This sends the leftmost value to the worker/main on the left and
            recieves its rightmost value. If it is the master, it will
            not try to send to or recieve from its left neighbour.
        */
        if (my_rank != 0)
        {
            MPI_Send(&send_buf[0], 2, MPI_DOUBLE, my_left, t, MPI_COMM_WORLD);
            MPI_Recv(&recv_buf[0], bufferSize, MPI_DOUBLE, my_left, t,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_current[0] = recv_buf[1];
        }

        /*
            This sends the rightmost value to the worker/main to the right and
            recieves its leftmost value. If it is the rightmost worker, it will
            not try to send to or recieve from its right neighbour.
        */
        if (my_rank != num_tasks - 1)
        {
            MPI_Send(&send_buf[0], 2, MPI_DOUBLE, my_right, t, MPI_COMM_WORLD);
            MPI_Recv(&recv_buf[0], bufferSize, MPI_DOUBLE, my_right, t,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_current[processSize + 1] = recv_buf[0];
        }

        /*
            This calculates the values of the next array
        */
        for (int i = 1; i < processSize + 1; i++)
        {
            local_next[i] = 2 * local_current[i] - local_old[i] +
                            0.15 * (local_current[i - 1] -
                            (2 * local_current[i] - local_current[i + 1]));
        }

        /*
            This ensures i_0 and i_max remain zero.
        */
        if (my_rank == 0)
        {
            local_next[1] = 0;
        }
        if (my_rank == num_tasks - 1)
        {
            local_next[processSize] = 0;
        }

        /*
            This reassigns the arrays for the next time-step.
        */
        memcpy(&local_old[0], &local_current[0], (processSize + 2) * sizeof(double));
        memcpy(&local_current[0], &local_next[0], (processSize + 2) * sizeof(double));
    }
}

/*
    This executes the entire simulation. It initializes the workers and
    contains the code for the master. It distributes the data and collecy=ts
    it at the end.
*/
double *simulate(const int i_max, const int t_max, double *old_array,
                 double *current_array, double *next_array)
{
    /*
        This initializes the processes and gives every process its rank.
    */
    int rc, num_tasks, my_rank;
    rc = MPI_Init(NULL, NULL);
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, " Unable to set up MPI ");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    workerSize = i_max / num_tasks;

    /*
        This is the master process. It will divide the data from the arrays
        and collect them at the end. It will also compute the first part of
        the array.
    */
    if (my_rank == 0)
    {
        /*
            The master computes the equation for the part of the array
            that is left over.
        */
        int masterSize = i_max - (num_tasks - 1) * workerSize;
        int start;

        /*
            The arrays are 2 larger than the masterSize. Only the last index
            is used as a halo cell, but this way it can use the compute
            function.
        */
        local_old = malloc((masterSize + 2) * sizeof(double));
        local_current = malloc((masterSize + 2) * sizeof(double));
        local_next = malloc((masterSize + 2) * sizeof(double));
        send_buf = malloc(workerSize * sizeof(double));
        recv_buf = malloc(workerSize * sizeof(double));

        /*
            This initializes its local arrays.
        */
        memcpy(&local_old[1], &old_array[0], masterSize * sizeof(double));
        memcpy(&local_current[1], &current_array[0], masterSize * sizeof(double));

        /*
            This sends a part of the old_array and current_array with
            size workerSize to worker i.
        */
        for (int i = 1; i < num_tasks; i++)
        {
            start = masterSize + (i - 1) * workerSize;
            memcpy(&send_buf[0], &old_array[start], workerSize * sizeof(double));
            MPI_Send(&send_buf[0], workerSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            memcpy(&send_buf[0], &current_array[start], workerSize * sizeof(double));
            MPI_Send(&send_buf[0], workerSize, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
        /*
            This computes the results and sends the values on the edges
            to other processes.
        */
        compute(workerSize, masterSize, num_tasks, my_rank, t_max);

        /*
            This collects the end results from the workers and sets them in
            current_array.
        */
        memcpy(&current_array[0], &local_current[1], masterSize * sizeof(double));
        for (int i = 1; i < num_tasks; i++)
        {
            start = masterSize + (i - 1) * workerSize;
            MPI_Recv(&recv_buf[0], workerSize, MPI_DOUBLE, i, i, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            memcpy(&current_array[start], &recv_buf[0], workerSize * sizeof(double));
        }
    }

    if (my_rank != 0)
    {
        worker(my_rank, num_tasks, i_max, t_max);
    }

    free(local_old);
    free(local_current);
    free(local_next);
    free(send_buf);
    free(recv_buf);

    MPI_Finalize();

    /*
        Only the master returns current_array.
    */
    if (my_rank == 0)
    {
        return current_array;
    }
    else
    {
        return NULL;
    }
}