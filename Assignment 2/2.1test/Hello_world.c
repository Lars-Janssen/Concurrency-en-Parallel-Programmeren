#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main ( int argc , char * argv [])
{
    int rc , num_tasks , my_rank ;
    rc = MPI_Init(&argc , &argv);                  // Init runtime
    if ( rc != MPI_SUCCESS) {                          // Success check
        fprintf (stderr , " Unable to set up MPI ");
        MPI_Abort(MPI_COMM_WORLD , rc );              // Abort runtime
    }
    MPI_Comm_size(MPI_COMM_WORLD, & num_tasks);     // Get num tasks
    MPI_Comm_rank(MPI_COMM_WORLD, & my_rank);       // Get task id
    printf ("Hello World says %s !\n" , argv[0]);
    printf ("I’m task number %d of a total of %d tasks.\n",
    my_rank , num_tasks);
    MPI_Finalize ();                                    // Shutdown runtime
    return 0;
}