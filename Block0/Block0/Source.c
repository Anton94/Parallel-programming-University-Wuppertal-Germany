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

// Task 3.
// Implement the Simpson`s rule with n subintervals for calculating.
// s = h / 6 * SUM(i=1..n-1)[f(xi) + 4f(xi+0.5) + f(xi+1)]
// where h is (b-a)/n and xi+0.5 = (xi + xi+1)/2
void task3(int argc, char * argv[])
{
	// TODO - broadcast all values in array...

	MPI_Init(&argc, &argv);

	int rank, processesCount;
	double a, b, h;
	double procSum, sum;
	double procInterval; // The subintervals size(number of calculations) for every process.
	int n, position;
	char buffer[100]; // Buffer and position for packing 'a', 'b' and 'n' data and broadcast them.
	double t1, t2; // Time measurement

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processesCount);


	if (rank == 0) // First one has to read the interval and number of subintervals. (a, b and n)
	{
		printf("Enter the interval bounds[a, b]: ");
		fflush(stdout);
		fscanf(stdin, "%lf %lf", &a, &b);
		printf("Enter the number of subintervals(n): ");
		fflush(stdout);
		fscanf(stdin, "%d", &n);

		t1 = MPI_Wtime();
		
		// Pack the data
		position = 0; // Position starts at the beginning of the @buffer.
		MPI_Pack(&a, 1, MPI_DOUBLE, buffer, 42, &position, MPI_COMM_WORLD);
		// The position has been incremented by sizeof(float) bytes (from MPI_Pack)
		MPI_Pack(&b, 1, MPI_DOUBLE, buffer, 42, &position, MPI_COMM_WORLD);
		MPI_Pack(&n, 1, MPI_INT, buffer, 42, &position, MPI_COMM_WORLD);

		// Now broadcast
		MPI_Bcast(buffer, 42, MPI_PACKED, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(buffer, 42, MPI_PACKED, 0, MPI_COMM_WORLD);

		// Let's unpack the buffer.
		position = 0;
		MPI_Unpack(buffer, 42, &position, &a, 1, MPI_DOUBLE, MPI_COMM_WORLD); // Once again the position is auto increment by sizeof(float) bytes.
		MPI_Unpack(buffer, 42, &position, &b, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(buffer, 42, &position, &n, 1, MPI_INT, MPI_COMM_WORLD);
	}

	// Broadcast a, b and n. They are 2 doubles and one integer. 
	// I do it once so it`s better to PACK them(not to make MPI struct) and send them with only one broadcast!
	//MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Calculate h and subinterval`s size.
	h = (b - a) / (double)n;
	double hPower16 = h;
	hPower16 *= hPower16 *= hPower16 *= hPower16 *= hPower16;

	procInterval = n / processesCount;
	if (n % processesCount != 0)
		procInterval += 1; // So the firs processesCount - 1 will take one more calculation and last one- the rest calculations

	procSum = 0.0;
	int i = rank * procInterval;
	int end = (i + procInterval < n) ? i + procInterval : n;
	double xCurr, xNext, xMid, xCurrPower, xNextPower;
	xCurr = a + i * h;
	for (; i < end; ++i)
	{
		xNext = xCurr + h;
		xMid = (xCurr + xNext) * 0.5;
		// TODO : x^16 better way
		xCurrPower = xCurr;
		xNextPower = xNext;
		xCurrPower *= xCurrPower *= xCurrPower *= xCurrPower *= xCurrPower;
		xMid *= xMid *= xMid *= xMid *= xMid;
		xNextPower *= xNextPower *= xNextPower *= xNextPower *= xNextPower;

		procSum += xCurrPower + 4 * xMid + xNextPower;

		xCurr = xNext;
	}

	// Reduce all particial sums
	MPI_Reduce(&procSum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// Synchonization point, no need
	// MPI_COMM_WORLD_Barrier();

	// Print proper output from the calculations.
	if (rank == 0)
	{
		t2 = MPI_Wtime();

		sum *= (h / 6.0);
		printf("Simpsons Rule for [%lf, %lf] with %d subintervals\n\tyielded the approximation %lf\n\tand took %.9f seconds(maybe)!", a, b, n, sum, t2 - t1);
	}

	MPI_Finalize();
}

int main(int argc, char* argv[])
{	
	task3(argc, argv);

	return 0;
}