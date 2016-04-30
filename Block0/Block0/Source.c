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

	printf("Program running on %d processes, I am no process. %d", processesCount, rank);

	MPI_Finalize();	
}

// Task 2.
// In process 0 set the user input.
// Then transpit it through the process ring to all other processes, while
// in each process double the value and output it.
void task2(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank, processesCount;
	double value, computed;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processesCount);
	
	if (rank == 0) // First one has to read the value from the user and send it to second one.
	{	
		printf("Enter a number: ");
		fflush(stdout);
		fscanf(stdin, "%lf", &value);
	}
	else // Every other process(exept 0) receives a value.
	{
		MPI_Recv(&value, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
	}

	computed = value * 2;
	printf("Process %d got %f and computed %f\n", rank, value, computed);

	if (rank < processesCount - 1) // every process, exept last one, sends a computed value.
	{
		MPI_Send(&computed, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

int main(int argc, char* argv[])
{	
	task2(argc, argv);

	return 0;
}