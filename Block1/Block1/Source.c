#include <stdlib.h>
#include "mpi.h"
#include "TwoDimArrays.h"

struct ProcData
{
	int rank, p;
	// Each processor has to allocate memory for it`s columns.
	// I will store them in one dim array - first column after that second and so on.
	// Another approach is with 2D array, but we will see which one is better. (iteration is not so good)
	double * columnsData;
	int dataCount; // Number of cells in all columns.
	//int dims[2];
	int M, N;
};


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

void allocateOneDimArrayForMultipleColsOfGivenProc(struct ProcData * procData)
{
	procData->dataCount = getTheTotalNumberOfElementsInProcColumns(procData->rank, procData->p, procData->M, procData->N);

	procData->columnsData = (double*)malloc(procData->dataCount * sizeof(double));
	// if (!columnsData)
	// error ....
}

void fillDataOfOneDimColumnsArray(const struct Matrix * matrix, double * data, int rank, int p, int M, int N)
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
}

// Distributes columns of MxN matrix over the processors (processor j holds column i if j === i mod p)
// @M is the number of rows and @N is the number of columns.
// Returns pointer to the new allocated memory for the columns...
void distributeColumns(const struct Matrix* matrix, struct ProcData * procData)
{
	// Each processor has to allocate memory for it`s columns.
	// I will store them in one array - first column after that second and so on.
	// Another approach is like 2D array, but we will see which one is better. (iteration is not so good)
	allocateOneDimArrayForMultipleColsOfGivenProc(procData);

	// I chose the 0 processor to distribute the matrix
	if (procData->rank == 0)
	{
		// For rank 0 I have all the data so simply write it.
		fillDataOfOneDimColumnsArray(matrix, procData->columnsData, procData->rank, procData->p, procData->M, procData->N);

		double * tempColumnsData = NULL;
		int tempDataCount = procData->dataCount; // Like proc 0

		// Send the data to each process
		int i;
		for (i = 1; i < procData->p; ++i)
		{
			// I want to reuse the allocated memory if possible.
			// If the current processor has the same number of columns like the previous one, simply keep it.
			// If there is allocated memory AND the previous procs has an extra column like the current one OR the prevous has no extra columns like the current one
			// I will keep the allocated memory, otherwise frees it and allocate new one.
			if (!(tempColumnsData && ((i - 1 < matrix->M % procData->p && i < matrix->M % procData->p) 
									|| (i - 1 >= matrix->M % procData->p && i >= matrix->M % procData->p)))
				)
			{
				free(tempColumnsData); // Free the memory
				tempDataCount = getTheTotalNumberOfElementsInProcColumns(i, procData->p, procData->M, procData->N);
				tempColumnsData = (double*)malloc(tempDataCount * sizeof(double)); // Allocate the new array
			}

			fillDataOfOneDimColumnsArray(matrix, tempColumnsData, i, procData->p, procData->M, procData->N); // Fill the data

			MPI_Send(tempColumnsData, tempDataCount, MPI_DOUBLE, i, 42, MPI_COMM_WORLD);
		}

		free(tempColumnsData);
	}
	else // Receive the data from process 0.
	{
		MPI_Recv(procData->columnsData, procData->dataCount, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
	}
}

// Collects the columns of all processors and processor 0 writes it to the given matrix.
void selectColumns(const struct Matrix* matrix, struct ProcData * procData)
{
	if (procData->rank == 0)
	{
		// Create temp buffer to get the data from all processors and to fill the matrix.
		double * tempBuffer = (double*)malloc(procData->dataCount * sizeof(double));
		double * pTempBuffer;
		int tempReceivedSize;
		// Write own data to the matrix.
		for (int i = 0; i < procData->p; ++i)
		{
			if (i == 0)
			{
				pTempBuffer = procData->columnsData;
			}
			else
			{
				tempReceivedSize = getTheTotalNumberOfElementsInProcColumns(i, procData->p, procData->M, procData->N); // Proc 0 has most(or equals to) other columns
				MPI_Recv(tempBuffer, tempReceivedSize, MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
				pTempBuffer = tempBuffer;
			}
			int k, j;
			for (k = i; k < matrix->M; k += procData->p) // Write the data to all k rows where k belongs to i-th processor
			{
				for (j = 0; j < matrix->N; ++j) // Write all data.
				{
					matrix->matrixData[k][j] = *pTempBuffer;
					++pTempBuffer;
				}
			}
		}

		free(tempBuffer);
	}
	else
	{
		MPI_Send(procData->columnsData, procData->dataCount, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
	}
}

// Returns the total number of columns I hold.
int transposeColumnsIHold(struct ProcData* procData)
{
	return procData->dataCount / procData->N;
}

// Returns the ammount of data to be send from @procData->rank processor to @toRank processor.
int transposeGetNumberOfEntriesToSendToProc(struct ProcData* procData, int toRank, int columnsIHold)
{
	// (the number of columns I hold) * (the number of rows for procees @toRank)
	int numberOfRowsForOtherProcess = procData->N / procData->p;
	if (toRank < procData->N % procData->p) // If it has extra row to send
		++numberOfRowsForOtherProcess;

	return columnsIHold * numberOfRowsForOtherProcess;
}

// Returns the number of rows given processor needs to receive from transposing.
int transposeRowsToReceive(struct ProcData* procData)
{
	int rowsToReceive = (procData->N / procData->p);
	if (procData->rank < procData->N % procData->p) // If it has extra row to receive
		++rowsToReceive;

	return rowsToReceive;
}

// Returns the ammount of data @procData->rank processor to receive from @toRank processor.
int transposeGetNumberOfEntriesToReceivFromProc(struct ProcData* procData, int fromRank, int rowsToReceive)
{
	// (the number of rows I have to receive) * (the number of columns @toRank holds)

	int numberOfColumnsOtherProcHolds = procData->M / procData->p;
	if (fromRank < procData->M % procData->p) // If it hold an extra column
		++numberOfColumnsOtherProcHolds;

	return rowsToReceive * numberOfColumnsOtherProcHolds;
}

// Transpose the matrix over each processor. A lot of communications(p^2)
void transpose(struct ProcData* procData)
{
	int newM = procData->N;
	int newN = procData->M; 

	// I have to send in total @p rows of data, one raw to each processor(exept mine..)
	// When matrix is transposed the rows become columns, so this processor holds every @p entry from a
	// matrix row so it has to send it to the processor which will hold the new column(which is the row in not transposed matrix)
	// in transposed matrix.
	// So, it has to send maximum of @p entris on one row, but it has every @p-th row data too, so in
	// total every @p entry on every @p-th row of the matrix. And this is for one processor.
	// Note: the sequence is first put the data from first needed row, after that the data for the 
	// next needed row(p-th row from first one) and so on.

	int k, i, j;

	struct Variable2DArray dataToSend, dataToReceive;
	dataToSend.ROWS = dataToReceive.ROWS = procData->p;
	dataToSend.rowSizes = (int*)malloc(dataToSend.ROWS * sizeof(int));	
	dataToReceive.rowSizes = (int*)malloc(dataToReceive.ROWS * sizeof(int));
	// Precalculate the total rows count that the processor need to receive and the columns it holds.
	int columnsIHold = transposeColumnsIHold(procData),
		rowsToReceive = transposeRowsToReceive(procData);
	for (k = 0; k < procData->p; ++k)
	{
		// Calculate the data count that needs to be send to k-th processor.
		dataToSend.rowSizes[k] = transposeGetNumberOfEntriesToSendToProc(procData, k, columnsIHold);
		// Calculate the data count that needs to be receive from k-th processor.
		dataToReceive.rowSizes[k] = transposeGetNumberOfEntriesToReceivFromProc(procData, k, rowsToReceive);
	}

	// Allocate the memory.
	variable2DArrayAllocate(&dataToSend);
	variable2DArrayAllocate(&dataToReceive);

	// Fill the data in the send matrix(Note: each row holds the data for correspondig processor, e.g. 2nd row for processor 2..)
	double * pRowData;
	for (k = 0; k < procData->p; ++k)
	{
		if (k == procData->rank) // Writes it directly in receive buffer(2D array)
			pRowData = dataToReceive.arrayData[k];
		else
			pRowData = dataToSend.arrayData[k];

		// Each @p-th element on each @p-th row starting from k-th (basicaly each element needed for processor @k)	
		for (i = k; i < procData->N; i += procData->p)
		{
			// j starts from i-th row and iterates by one column size 
			for (j = i; j < procData->dataCount; j += procData->N) // Note: I keep it transposed, so N is the number of rows(the size of one column)!
			{
				*pRowData++ = procData->columnsData[j];
			}
		}			
	}

	// Send each row of the matrix to the corespondig processor.
	// Receive a row of data from the coresponding processor.
	// Other approach for k = 0 to procData->rank && from procData->rank + 1 to p
	for (k = 0; k < procData->p; ++k)
	{
		int next = (procData->rank + k) % procData->p,
			prev = (procData->rank + procData->p - k) % procData->p;
		
		// TODO deadlock has here...

		// I don`t want to send to me and receive from me.
		if (procData->rank % 2 == 1) 
		{
			if (procData->rank != next)
			{
				MPI_Send(dataToSend.arrayData[next], dataToSend.rowSizes[next], MPI_DOUBLE, next, 42, MPI_COMM_WORLD);
				MPI_Recv(dataToReceive.arrayData[prev], dataToReceive.rowSizes[prev], MPI_DOUBLE, prev, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			}
		}
		else if (prev != procData->rank)
		{
			MPI_Recv(dataToReceive.arrayData[prev], dataToReceive.rowSizes[prev], MPI_DOUBLE, prev, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			MPI_Send(dataToSend.arrayData[next], dataToSend.rowSizes[next], MPI_DOUBLE, next, 42, MPI_COMM_WORLD);
		}
	}

	// Now lets create the new data of processor procData->rank.
	procData->M = newM;
	procData->N = newN;
	// Free the old data.
	free(procData->columnsData);
	// Allocate the new memory for the new data.
	allocateOneDimArrayForMultipleColsOfGivenProc(procData);

	// Create extra pointers to each row of received 2D array so I can iterate on them
	double ** pArrayData = (double**)malloc(dataToReceive.ROWS * sizeof(double*));
	for (i = 0; i < procData->p; ++i)
	{
		pArrayData[i] = dataToReceive.arrayData[i];
	}

	double * pProcData = procData->columnsData; // For iteration on columnsData.

	// Fill the data in the new memory
	int idx;
	int columnsItHolds = procData->dataCount / procData->N;
	for (i = 0; i < columnsItHolds; ++i) // For each column it hold(in not transposed row it hold)
	{
		// Goes through the whole column (in not transposed row) and writes the data cycling from each other processor.
		for (j = 0; j < procData->N; ++j)
		{
			*(pProcData)++ = *pArrayData[j % procData->p]++;
		}
	}

	free(pArrayData);
	variable2DArrayFree(&dataToSend);
	variable2DArrayFree(&dataToReceive);
}

// Deletes the allocated memory for the processor columns.
void freeProcessAllocatedMemory(struct ProcData * procData)
{
	free(procData->columnsData);
	procData->dataCount = 0;
}

/// Main functionality.
// runs the functionality for matrix with @ROWS number of rows, @COLS number of columns and
// if outputing is TRUE it prints the matrixes and 
// adds the time for distribute, transpose and select to @tSum EXEPT the time for creating and freeing the matrixes. (only inportant for @rank = 0
int functionality(struct ProcData * procData, int ROWS, int COLS, int outputing, double * tSum)
{
	struct Matrix matrixToSend, matrixToReceive; // Only used in processor with rank 0.
	int dims[2];

	if (procData->rank == 0)
	{
		dims[0] = ROWS;
		dims[1] = COLS;
	}

	MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
	// Keep the data transposed.
	procData->M = dims[1];
	procData->N = dims[0];

	/* Create the matrixes in proc 0 */
	// It`s C , so I will keep the matrix 'transposed', because I want to send whole columns,
	// So I will keep the matrix as rows of columns(first row is the first column, second row is the second column and so on).
	if (procData->rank == 0)
	{
		// Generate matrixes
		matrixAllocate(&matrixToSend, procData->M, procData->N);	 // NOTE: here I give transposed matrix, so swaped number of rows and cols.
		matrixAllocate(&matrixToReceive, procData->N, procData->M);// Transposed one but the returned is also transposed

		matrixSetDefaultValues(&matrixToSend);

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

	// Transpose the columns data.
	transpose(procData);
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

// Testing the functionality for multiple matrixes with row sizes from [a, b] and column sizes from [c, d]
void test(int a, int b, int c, int d, int outputingTestStatus, int outputingMatrixValues)
{
	struct ProcData procData;

	MPI_Comm_rank(MPI_COMM_WORLD, &procData.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procData.p);

	double tSum = 0.0;

	int i, j;
	int res;
	int failedTests = 0, totalTests = (b - a + 1) * (d - c + 1); // Because it`s closed interval.
	for (i = a; i <= b; ++i)
	{
		for (j = c; j <= d; ++j)
		{
			res = functionality(&procData, i, i, outputingMatrixValues, &tSum);
			if (procData.rank == 0)
			{
				if (!res)
					++failedTests;
				if (outputingTestStatus)
				{
					printf("Testing with matrix %d-by-%d...\n", i, j);
					if (res)
						printf("\t ...success!\n");
					else
						printf("\t ...failed!\n");
				}
			}
		}
	}	

	if (procData.rank == 0)
	{
		printf("Testing with matrix with rows from [%d, %d] and columns from [%d, %d] \n\t...took %.9f seconds!\n", a, b, c, d, tSum);
		printf("\t...with %d tests failed out of %d!\n", failedTests, totalTests);;
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	//test(1000, 1100, 1000, 1100, 0, 0);
	test(1, 150, 1, 150, 0, 0);
	//test(200, 290, 200, 290, 1, 0);
	//test(220, 240, 220, 240, 1, 0);
//	test(11, 11, 9, 9, 1, 1);

	MPI_Finalize();

	return 0;
}