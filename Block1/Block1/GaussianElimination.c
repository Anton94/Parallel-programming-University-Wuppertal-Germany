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
	if (i > matrix->M || j > matrix->N || i < 0 || j < 0)
		return -1.1;
	if (i < matrix->N)  // If the entry is for the original matrix A.
		return ((double)(i + j) / (double)matrix->N) * sin((double)i*(double)j*PI / (matrix->N + 1));
	else // The entry is from the original matrix B.
		return i - matrix->N + j; // i + j, but @i "starts" from N.
}

// Simple swap with third variable.
void swap(double *a, double *b)
{
	double temp = *a;
	*a = *b;
	*b = temp;
}

// Execute kji column pivoting parallel Gaussian elimination.
// Each processor has it's columns in the original matrix, one after another in a vector.
// Here it may be better to use 2D vector, instead of 1D.
void gaussianElimination(struct ProcData* procData)
{
	int k, j, i,
		pivotIndex,
		mineColumnIndex,
		root;

	double * myData = procData->columnsData,
		*pEndData;

	double * l = (double*)malloc(procData->N * sizeof(double)); // Allocate temp buffer for the multipliers.

	for (k = 0; k < procData->N; ++k)
	{
		root = k % procData->p;
		if (procData->rank == root)
		{
			mineColumnIndex = k / procData->p; // Determine which of my columns is this column @k
			pivotIndex = k;
			myData += mineColumnIndex*procData->N; // Start from the column K (which is (k/p)-th of mine columns)

			// Column pivoting.
			for (i = k + 1; i < procData->N; ++i)
			{
				if (myData[i] > myData[pivotIndex])
					pivotIndex = i;
			}

			myData = procData->columnsData + mineColumnIndex*procData->N; //TODO

			// Calculate the multipliers.
			for (i = k + 1; i < procData->N; ++i)
			{
				l[i] = myData[i] = myData[i] / myData[k];
			}
		}

		MPI_Bcast(&pivotIndex, 1, MPI_INT, root, MPI_COMM_WORLD);

		// Interchange data of rows @pivotIndex and @k.
		if (pivotIndex != k)
		{
			myData = procData->columnsData;
			for (i = procData->rank; i < procData->M; i += procData->p)
			{
				swap(&myData[i], &myData[pivotIndex]);
				myData += procData->N;
			}
		}

		if (k + 1 < procData->N)
		{
			// Distribute the multipliers. L[k+1:N, k]
			MPI_Bcast(l + k + 1, procData->N - k - 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

			// Update the trailing submatrix.

			// Determin the starting column (of K+1xK+1 trailing matrix) of the curren processor.
			mineColumnIndex = procData->rank + procData->p / (k + 1); // It starts from @rank position.
			if (mineColumnIndex * procData->p < k + 1)
				++mineColumnIndex;
			pEndData = procData->columnsData + mineColumnIndex * procData->N;
			while (myData < pEndData) // While I have more columns to update
			{
				for (i = k + 1; i < procData->N; ++i)
				{
					myData[i] = myData[i] - l[i] * myData[k];
				}
				myData += procData->N;
			}
		}		
	}
	//TODO FIX ERROR
	free(l);
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
		matrixAllocate(&matrixToReceive, procData->M, procData->N); 

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

	// Execute the Gaussian elimination.
	gaussianElimination(procData);
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

		// TODO: Check if possible...
		int res = 1;

		t2 = MPI_Wtime();
		*tSum += t2 - t1;
		matrixFree(&matrixToReceive);
		matrixFree(&matrixToSend);
		return res;
	}

	return -1; // All other processors don`t care the return value.
}