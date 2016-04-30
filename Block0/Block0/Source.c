//#include "stdafx.h"
#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

// Task 1.
// For each proccess, determenate it`s number and prints out proper output
void task1(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int rank, processesCount;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processesCount);
	printf("Program running on %d processes, I am no process. %d", processesCount, rank);/*
	if (rank == 0)
	{
		char helloStr[] = "Hello world";
		MPI_Send(helloStr, _countof(helloStr), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
	}
	else if (rank == 1)
	{
		char helloStr[12];
		MPI_Recv(helloStr, _countof(helloStr), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		printf("Rank 1 received string %s from Rank 0\n", helloStr);
	}*/

	MPI_Finalize();
	
}

int main(int argc, char* argv[])
{
	
	task1(argc, argv);

	return 0;
}
