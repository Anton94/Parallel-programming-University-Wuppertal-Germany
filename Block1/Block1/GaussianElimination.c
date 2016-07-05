#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "GaussianElimination.h"

#define PI 3.141592653589793238462643383

/*
 * The Matrix (A|B) has the memory representation as (N+Q)xN , because each row of the matrix represents one column of the (A|B)
 * transposed...
 */

// Returns the entry value of the matrix on row @i and column j.
// The default values for the TRANSPOSED MATRIX (A|B)
// Are, for first N rows(which are first N columns of the original matrix A)
// Aij = (i + j)/N * sin[(i*j*pi)/(N + 1)]
// And for the B part of (A|B) (transposed, so, it's columns are the rows: N+1..M(M=N+Q))
// Bij = i + j
double getGEEntryValue(struct Matrix * matrix, int i, int j)
{
	if (i < matrix->N)  // If the entry is for the original matrix A.
		return ((double)(i + j) / (double)matrix->N) * sin((double)i*(double)j*PI / (matrix->N + 1));
	else // The entry is from the original matrix B.
		return i - matrix->N + j; // i + j, but @i "starts" from N.
}

/// Main functionality.
// runs the functionality for matrix A with dimension sizes @dimA - number of rows(and same number of columns)
// @colsB number of columns of matrix B.
// If outputing is TRUE it prints the matrixes and 
// adds the time for distribute, transpose and select to @tSum EXEPT the time for creating and freeing the matrixes. (only inportant for @rank = 0
int functionalityGE(struct ProcData * procData, int dimA, int colsB, int outputing, double * tSum)
{
	struct Matrix matrixToSend, matrixToReceive; // Only used in processor with rank 0.
	int dims[2];

	if (procData->rank == 0)
	{
		dims[0] = dimA;
		dims[1] = colsB;
	}

	MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
	// Keep the data transposed.
	procData->M = dims[0] + dims[1]; // Total number of columns(exacly the number of rows in the transposed matrix, which I use)..
	procData->N = dims[0]; // Number of entries in each column(exacly the number of columns in the transposed..)

	/* Create the matrixes in proc 0 */
	// It`s C , so I will keep the matrix 'transposed', because I want to send whole columns,
	// So I will keep the matrix as rows of columns(first row is the first column, second row is the second column and so on).
	if (procData->rank == 0)
	{
		// Generate matrixes
		matrixAllocate(&matrixToSend, procData->M, procData->N);	 // NOTE: here I give transposed matrix, so swaped number of rows and cols.
		matrixAllocate(&matrixToReceive, procData->N, procData->M);// Transposed one but the returned is also transposed

		matrixSetValues(&matrixToSend, &getGEEntryValue);

		if (outputing)
		{
			printf("Matrix:\n");
			matrixPrintTransposed(&matrixToSend);
		}
	}

	double t1, t2;

	t1 = MPI_Wtime();

	// Now lets distribute the matrix.(which is stored by columns) // Transposed dimentions COLS <-> ROWS
	distributeColumns(&matrixToSend, procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	// Get the matrix data from all processes.
	selectColumns(&matrixToReceive, procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	// Free the memory allocated for the columns.
	freeProcessAllocatedMemory(procData);
	if (procData->rank == 0)
	{
		if (outputing)
		{
			printf("Received matrix:\n");
			matrixPrintTransposed(&matrixToReceive);
		}

		// Compare and check if the received one is correct.
		int res = matrixCompareWithOtherMatrix(&matrixToSend, &matrixToReceive);

		t2 = MPI_Wtime();
		*tSum += t2 - t1;
		matrixFree(&matrixToReceive);
		matrixFree(&matrixToSend);
		return res;
	}

	return -1; // All other processors don`t care the return value.
}