#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    pid_t pid = getpid();
    printf("Loop PID: %d\n", pid);

    // Convert to int64_t
    int64_t pid_int64 = (int64_t)pid;

    // Open a file for writing
    FILE *file = fopen("benchmarking/pids/loop.pid", "wb");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    // Write the int64_t value to the file
    size_t num_written = fwrite(&pid_int64, sizeof(int64_t), 1, file);
    if (num_written != 1)
    {
        perror("Error writing to file");
        fclose(file);
        return 1;
    }

    fclose(file);

    printf("PID written to benchmarking/pids/loop.pid\n");

    double start, end;
    double runTime;
    int num = 1, primes = 0;

    int limit = 10000000;

#pragma omp parallel for schedule(dynamic) reduction(+ : primes)
    for (num = 1; num <= limit; num++)
    {
        int i = 2;
        while (i <= num)
        {
            if (num % i == 0)
                break;
            i++;
        }
        if (i == num)
            primes++;
        //      printf("%d prime numbers calculated\n",primes);
    }

    runTime = end - start;
    printf("This machine calculated all %d prime numbers under %d in %g seconds\n", primes, limit, runTime);

    remove("../pids/pid-loop.txt");

    return 0;
}
