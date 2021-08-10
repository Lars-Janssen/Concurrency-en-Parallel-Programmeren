/*
    Aron de Ruijter 12868655, Lars Janssen 12882712

    This program searches for prime numbers with the sieve of Eratosthenes.
    The basic premise of this algorithm is that it stars with every natural
    larger than 1. Then it says the first number in the list is prime and
    removes every multiple of it. Then it again says the first number in the
    list is prime and removes every multiple of it, etc.
    For more info, check https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes.

    In this program, we try to parallelize this process by having a thread for
    every prime we want to filter. We then create the list of numbers and
    feed it from thread to thread, with each thread narrowing the list down.

    So we start by making a thread to filter the multiples of 2. When we get to
    the next item that is not filtered away, we make a new thread with this
    item (in this case 3) as filter and feed every item that is not divisible
    by two to this thread. We repeat this making of thread until we have the
    number of primes desired.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include "thread.h"

/*
    The maximum number of items that can be in a queue.
*/
#define MAX 10


/*
    This specifies how many primes we want to find and makes a counter.
*/
int nth_number = 1000;
int prime = 0;

/*
    This struct is used for the input- and outputqueue. My queue is based on
    this design: https://www.programiz.com/dsa/circular-queue.
    It keeps track of the front of the queue, where we should take items from,
    the back, where we should place items, and the number of items. It stores
    the queue in the array.
*/
typedef struct
{
    int front;
    int back;
    int items;
    int array[MAX];
} queue;

/*
    conditions and a lock to ensure thread safety.
*/
pthread_cond_t full = PTHREAD_COND_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock;

/*
    This function is a new thread in the pipeline. It will take items from
    the inputqueue, see if it is a multiple of the filter, and if not,
    set it in the outputqueue.
*/
void *thread(void *a)
{
    /*
        Increase the prime count and set the filter for this thread.
        The filter is the integer at the front of the queue.
    */
    int newThread = 0;
    prime++;

    queue *input = (queue*) a;
    int current = input->array[input->front];
    int filter = current;
    printf("%d\n", filter);

    queue output;
    output.front = 0;
    output.back = -1;
    output.items = 0;

    pthread_t thread_id;

    while(filter > 0)
    {
        /*
            We check if if the input is empty, and wait if it is.
        */
       pthread_mutex_lock(&lock);
       while(input->items <= 0)
       {
            pthread_cond_wait(&empty, &lock);
       }

        current = dequeue(input);

        pthread_cond_signal(&full);
        pthread_mutex_unlock(&lock);

        /*
            We check if an integer needs to be send through, and wait if
            the outputqueue is full.
        */
        if(current % filter != 0 && current != -1)
        {
            pthread_mutex_lock(&lock);
            while (output.items >= MAX)
            {
                pthread_cond_wait(&full, &lock);
            }

            enqueue(&output, current);

            pthread_cond_signal(&empty);
            pthread_mutex_unlock(&lock);

            /*
                If this is the first integer not filtered, then
                we make a new thread with it as filter.
            */
            if(newThread == 0)
            {
                pthread_create(&thread_id, NULL, thread, (void*)&output);
                newThread = 1;
            }
        }
    }
    return NULL;
}

/*
    This function adds an item to the outputqueue.
*/
void *enqueue(void *a, int item)
{
    queue *output = (queue*) a;
    output->back = (output->back + 1) % MAX;
    output->array[output->back] = item;
    output->items++;
    return NULL;
}

/*
    This function removes an item from the inputqueue.
*/
int dequeue(void *a)
{
    queue *input = (queue*) a;
    if(input->items > 0)
    {
        int item = input->array[input->front];
        input->front = (input->front + 1) % MAX;
        input->items--;
        return item;
    }
    else
    {
        return -1;
    }

}

/*
    The main thread is used to keep the time and generate a sequence of
    natural numbers, which it uses as an outputqueue to send to the first
    thread.
*/
int main()
{
    printf("%dth prime \n", nth_number);

    pthread_mutex_init(&lock, NULL);

    /*
        This starts the clock.
    */
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    /*
        This integer is incremented to generate the natural numbers.
    */
    int integer = 2;

    /*
        This makes the outputqueue for the natural numbers(except for 1),
        which is the inputqueue for the filter of multiples of 2.
        It then initialises this filter thread with a pointer to this outputqueue.
    */
    pthread_t thread_id;
    queue output;
    output.front = 0;
    output.back = 0;
    output.items = 0;
    output.array[0] = 2;
    output.items++;
    pthread_create(&thread_id, NULL, thread, (void*)&output);

    /*
        This loop generates the natural numbers (except for 1).
    */
    while(integer > 0)
    {
        /*
            This checks if we have generated enough primes, and if so,
            stops the clock and prints the time taken.
        */
        if(prime >= nth_number)
        {
            pthread_mutex_destroy(&lock);
            clock_gettime(CLOCK_REALTIME, &end);
            long seconds = end.tv_sec - start.tv_sec;
            long nanoseconds = end.tv_nsec - start.tv_nsec;
            double elapsed = seconds + nanoseconds * 1e-9;
            printf("%.3f seconds \n", elapsed);
            exit(EXIT_SUCCESS);
        }

        /*
            This checks if there is room in the outputqueue, and waits if
            there is no room.
        */
        integer++;
        pthread_mutex_lock(&lock);
        while (output.items >= MAX)
        {
            pthread_cond_wait(&full, &lock);
        }

        enqueue(&output, integer);

        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&lock);

    }

    return EXIT_SUCCESS;
}