#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <unistd.h>

int main()
{
    int welcomeSocket, newSocket;
    char buffer[1024];
    struct sockaddr_in serverAddr;
    struct sockaddr_storage serverStorage;
    socklen_t addr_size;

    pid_t pid = getpid();
    printf("Server PID: %d\n", pid);

    // Convert to int64_t
    int64_t pid_int64 = (int64_t)pid;

    // Open a file for writing
    FILE *file = fopen("benchmarking/pids/server.pid", "wb");
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

    printf("PID written to benchmarking/pids/server.pid\n");

    /*---- Create the socket. The three arguments are: ----*/
    /* 1) Internet domain 2) Stream socket 3) Default protocol (TCP in this case) */
    welcomeSocket = socket(PF_INET, SOCK_STREAM, 0);

    /*---- Configure settings of the server address struct ----*/
    /* Address family = Internet */
    serverAddr.sin_family = AF_INET;
    /* Set port number, using htons function to use proper byte order */
    serverAddr.sin_port = htons(7891);
    /* Set IP address to localhost */
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    /* Set all bits of the padding field to 0 */
    memset(serverAddr.sin_zero, '\0', sizeof serverAddr.sin_zero);

    /*---- Bind the address struct to the socket ----*/
    bind(welcomeSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr));

    /*---- Listen on the socket, with 5 max connection requests queued ----*/
    if (listen(welcomeSocket, 5) == 0)
        printf("Listening\n");
    else
        printf("Error\n");

    /*---- Accept call creates a new socket for the incoming connection ----*/
    addr_size = sizeof serverStorage;
    newSocket = accept(welcomeSocket, (struct sockaddr *)&serverStorage, &addr_size);

    /*---- Send message to the socket of the incoming connection ----*/
    strcpy(buffer, "Hello World\n");
    send(newSocket, buffer, 13, 0);

    return 0;
}
