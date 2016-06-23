#include <stdlib.h>
#include "mpi.h"

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


/* Matrix struct .. basic.. */
struct Matrix
{
	int M, N; // Rows and cols number
	double ** matrixData;
};

// Generates new matrix by the given M and N.
// Need manualy to take care to delete the old matrix.
// Returns 0 if it manage to allocate the memory for the matrix and 0 if not.
int matrixAllocate(struct Matrix* matrix, int m, int n)
{
	matrix->M = m;
	matrix->N = n;

	matrix->matrixData = (double**)malloc(matrix->M * sizeof(double*)); // Allocate the vector which holds the pointers to diff columns
	if (!matrix->matrixData)
	{
		printf("Failed to allocate the matrix!\n");
		return 1;
	}
	for (int i = 0; i < matrix->M; ++i)
	{
		matrix->matrixData[i] = (double*)malloc(matrix->N * sizeof(double)); // Allocate the vector for the columns.
		if (!matrix->matrixData[i])
		{
			printf("Failed to allocate the matrix!\n");
			// Now free the allocated memory so far.
			for (int j = i - 1; j >= 0; --i)
				free(matrix->matrixData[i]);
			free(matrix->matrixData); // Free the array of pointers to the column arrays.
			matrix->matrixData = NULL;

			return 1;
		}
	}

	return 0;
}

// Frees the allocated memory for the matrix. Sets M and N to 0.
void matrixFree(struct Matrix* matrix)
{
	for (int i = 0; i < matrix->M; ++i)
		free(matrix->matrixData[i]);

	free(matrix->matrixData); // Free the array of pointers to the column arrays.

	matrix->M = matrix->N = 0;
	matrix->matrixData = NULL;
}

// Sets the default values for the matrix Aij = j + i / 1000 (transposed...)
void matrixSetDefaultValues(struct Matrix* matrix)
{
	if (matrix->matrixData == NULL)
		return; // A little safty..
	for (int i = 0; i < matrix->M; ++i)
	{
		double sum = i / 1000.0;
		for (int j = 0; j < matrix->N; ++j)
		{
			matrix->matrixData[i][j] = j + sum;
		}
	}
}

// Prints the matrix to the standard output.
void matrixPrint(const struct Matrix * matrix)
{
	if (matrix->matrixData == NULL)
		return; // A little safty..
	int maxWidth = 8;
	int i, j;
	for (i = 0; i < matrix->M; ++i)
	{
		for (j = 0; j < matrix->N; ++j)
		{
			printf("%*.3f ", maxWidth, matrix->matrixData[i][j]);
		}
		printf("\n");
	}
}

// Prints the matrix to the standard output TRANSPOSED.
void matrixPrintTransposed(const struct Matrix * matrix)
{
	if (matrix->matrixData == NULL)
		return; // A little safty..
	int i, j;
	for (j = 0; j < matrix->N; ++j)
	{
		for (i = 0; i < matrix->M; ++i)
		{
			printf("%4f ", matrix->matrixData[i][j]);
		}
		printf("\n");
	}
}


/* END of matrix stuff */

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

	//printf("%d has data count of %d\n", procData->rank, procData->dataCount);
	//printf("Data count of proc %d = %d\n", rank, dataCount);
	// I chose the 0 processor to distribute the matrix
	if (procData->rank == 0)
	{
		// For rank 0 I have all the data so simply write it.
		fillDataOfOneDimColumnsArray(matrix, procData->columnsData, procData->rank, procData->p, procData->M, procData->N);
	
		
		/*printf("Proc 0 has\n");
		
		int j;
		for (j = 0; j < dataCount; ++j)
		{
			printf("%.3f ", columnsData[j]);
		}
		printf("\n");*/



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
			if (!(tempColumnsData && ((i - 1 < matrix->N % procData->p && i < matrix->N % procData->p) 
									|| (i - 1 >= matrix->N % procData->p && i >= matrix->N % procData->p)))
				)
			{
				free(tempColumnsData); // Free the memory
				tempDataCount = getTheTotalNumberOfElementsInProcColumns(i, procData->p, procData->M, procData->N);
				tempColumnsData = (double*)malloc(tempDataCount * sizeof(double)); // Allocate the new array
			}

			fillDataOfOneDimColumnsArray(matrix, tempColumnsData, i, procData->p, procData->M, procData->N); // Fill the data
			
			/*int j;
			for (j = 0; j < tempDataCount; ++j)
			{
				printf("%.3f ", tempColumnsData[j]);
			}*/

			MPI_Send(tempColumnsData, tempDataCount, MPI_DOUBLE, i, 42, MPI_COMM_WORLD);
		}

		free(tempColumnsData);
	}
	else // Receive the data from process 0.
	{
		//printf("Receive %d in proc %d\n", *dataCount, rank);
		MPI_Recv(procData->columnsData, procData->dataCount, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		/*int j;
		for (j = 0; j < *dataCount; ++j)
		{
			printf("%.3f ", columnsData[j]);
		}
		printf("\n");*/
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
	//	printf("Data count %d\n", procData->dataCount);
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
			//	printf("Tries to recv %d from %d\n", tempReceivedSize, i);
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
	//	printf("%d tries to send %d\n", procData->rank, procData->dataCount);
		MPI_Send(procData->columnsData, procData->dataCount, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
	}
}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	
	struct ProcData procData;
	struct Matrix matrix; // Only used in processor with rank 0.
	
	
	// I make the number of rows and cols to be in all processors, other way is to broadcast them.
	// TODO Broadcast them!
	int ROWS = 14;
	int COLS = 8;

	procData.M = COLS;
	procData.N = ROWS;

	MPI_Comm_rank(MPI_COMM_WORLD, &procData.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procData.p);

	/* Create the matrix in proc 0 */
	// It`s C , so I will keep the matrix 'transposed', because I want to send whole columns,
	// So I will keep the matrix as rows of columns(first row is the first column, second row is the second column and so on).
	if (procData.rank == 0)
	{
		// Generate matrix 3x5, but transposed...
		if (matrixAllocate(&matrix, COLS, ROWS)) // NOTE: here I give transposed matrix, so swaped number of rows and cols.
			return -1; // TODO;
		matrixSetDefaultValues(&matrix);
		//matrixPrint(&matrix);
		matrixPrintTransposed(&matrix);
	}
	// Now lets distribute the matrix.(which is stored by columns) // Transposed dimentions COLS <-> ROWS
	distributeColumns(&matrix, &procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	if (procData.rank == 0) // Delete the matrix
	{
		matrixFree(&matrix);
		matrixAllocate(&matrix, COLS, ROWS); // Allocate memory for new matrix
	}

	//printf("%d has to send %d\n", procData.rank, procData.dataCount);
	// Get the matrix data from all processes.
	selectColumns(&matrix, &procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	/* TO DO synchronize if needed.*/
	if (procData.rank == 0)
	{
		printf("Received matrix:\n");
		matrixPrintTransposed(&matrix);
		matrixFree(&matrix);
	}

	MPI_Finalize();
	return 0;
}