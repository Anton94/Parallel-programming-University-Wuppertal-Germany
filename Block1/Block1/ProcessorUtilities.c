#include <stdlib.h>
#include "mpi.h"
#include "TwoDimArrays.h"
#include "ProcessorUtilities.h"

// Returns tohe total number of entries of the columns of the given processor rank(@rank).
// I don't use the structur ProcData, because it may be called for different processors.
int getTheTotalNumberOfElementsInProcColumns(int rank, int p, int M, int N)
{
	int dataCount = 0;
	// The matrix is transposed, so M rows , each row represent a column
	// j-th row is in @rank processor if rank === j mod p
	dataCount = M / p;
	// If there are more columns(N != q.p ; q some pos. number)
	if (rank < M % p)
		++dataCount;
	// Now dataCount has the number of columns, multiply it with N to get the number of cells.
	dataCount *= N;

	return dataCount;
}

// Allocates vector for the data of all columns of the given processor.
// Column one after that column two and so on...
// Also sets the total number of entries in all columns which are in the given processor.
void allocateOneDimArrayForMultipleColsOfGivenProc(struct ProcData * procData)
{
	procData->dataCount = getTheTotalNumberOfElementsInProcColumns(procData->rank, procData->p, procData->M, procData->N);

	procData->columnsData = (double*)malloc(procData->dataCount * sizeof(double));
	// if (!columnsData)
	// error ....
}

// I don't use ProcData structure because it's not necessary to fill own data, but for other processor.
// Returns a pointer to the cell which is one after the last writen value in the given vector @data.
double* fillDataOfOneDimColumnsArray(const struct Matrix * matrix, double * data, int rank, int p, int M, int N)
{
	int i, j;
	// Each row @i mod p == rank - write it's data to the array
	// Start from row @rank and increment the rows count by p, this are the proc @rank columns.
	for (i = rank; i < M; i += p)
	{
		for (j = 0; j < N; j++)
		{
			*data = matrix->matrixData[i][j];
			++data;
		}
	}

	return data;
}

// Deletes the allocated memory for the processor columns.
void freeProcessAllocatedMemory(struct ProcData * procData)
{
	free(procData->columnsData);
	procData->dataCount = 0;
}