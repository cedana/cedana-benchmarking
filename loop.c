#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>

void *pidUpdater(void *arg)
{
    FILE *file = (FILE *)arg;
    while (1)
    {
        pid_t pid = getpid();
        int64_t pid_int64 = (int64_t)pid;

        char filename[100]; // Adjust the size as needed
        sprintf(filename, "benchmarking/pids/loop-%lld.pid", (long long)pid_int64);

        FILE *pidFile = fopen(filename, "wb");
        if (pidFile == NULL)
        {
            perror("Error opening PID file");
            pthread_exit(NULL);
        }

        size_t num_written = fwrite(&pid_int64, sizeof(int64_t), 1, pidFile);
        if (num_written != 1)
        {
            perror("Error writing to PID file");
        }

        fclose(pidFile);

        chmod(filename, 0644);

        sleep(1);
    }
}

int main()
{
    pid_t pid = getpid();
    printf("Loop PID: %d\n", pid);

    int64_t pid_int64 = (int64_t)pid;

    char filename[100]; // Adjust the size as needed
    sprintf(filename, "benchmarking/pids/loop-%lld.pid", (long long)pid_int64);

    FILE *file = fopen(filename, "wb");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    double start, end;
    double runTime;
    int num = 1, primes = 0;

    int limit = 10000000;

    pthread_t pidThread;
    if (pthread_create(&pidThread, NULL, pidUpdater, (void *)file) != 0)
    {
        perror("Error creating PID thread");
        fclose(file);
        return 1;
    }

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
    }

    pthread_join(pidThread, NULL);

    fclose(file);

    return 0;
}
