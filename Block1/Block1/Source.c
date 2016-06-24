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

// Matrix with variable row lengths
struct Variable2DArray
{
	int ROWS;
	double ** arrayData;
	int * rowSizes;
};

// Generates new matrix by the given number of rows(@ROWS) and array with row sizes.
// Need manualy to take care to delete the old matrix.
// Need the ROWS and rowSizes array to be seted before calling 'variable2DArrayAllocate'!!!
// Returns 0 if it manage to allocate the memory for the matrix and 0 if not.
int variable2DArrayAllocate(struct Variable2DArray* variable2Darray)
{
	variable2Darray->arrayData = (double**)malloc(variable2Darray->ROWS * sizeof(double*)); // Allocate the vector which holds the pointers to diff rows
	if (!variable2Darray->arrayData)
	{
		printf("Failed to allocate the matrix!\n");
		return 1;
	}
	int i, j;
	for (i = 0; i < variable2Darray->ROWS; ++i)
	{
		variable2Darray->arrayData[i] = (double*)malloc(variable2Darray->rowSizes[i] * sizeof(double)); // Allocate the vector for the columns.
		if (!variable2Darray->arrayData[i])
		{
			printf("Failed to allocate the matrix!\n");
			// Now free the allocated memory so far.
			for (j = i - 1; j >= 0; --i)
				free(variable2Darray->arrayData[i]);
			free(variable2Darray->arrayData); // Free the array of pointers to the column arrays.
			variable2Darray->arrayData = NULL;

			return 1;
		}
	}

	return 0;
}

// Frees the allocated memory for the 2D array. Sets default values.
void variable2DArrayFree(struct Variable2DArray* variable2Darray)
{
	int i;
	for (i = 0; i < variable2Darray->ROWS; ++i)
		free(variable2Darray->arrayData[i]);

	free(variable2Darray->arrayData); // Free the array of pointers to the column arrays.
	free(variable2Darray->rowSizes);

	variable2Darray->ROWS = 0;
	variable2Darray->arrayData = NULL;
	variable2Darray->rowSizes = NULL;
}

// Prints the matrix to the standard output.
void variable2DArrayPrint(const struct Variable2DArray * variable2Darray)
{
	if (variable2Darray->arrayData == NULL)
		return; // A little safty..
	int i, j;
	for (i = 0; i < variable2Darray->ROWS; ++i)
	{
		for (j = 0; j < variable2Darray->rowSizes[i]; ++j)
		{
			printf("%*.3f ", 6, variable2Darray->arrayData[i][j]);
		}
		printf("\n");
	}
}


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

	matrix->matrixData = (double**)malloc(matrix->M * sizeof(double*)); // Allocate the vector which holds the pointers to diff rows
	if (!matrix->matrixData)
	{
		printf("Failed to allocate the matrix!\n");
		return 1;
	}
	int i, j;
	for (i = 0; i < matrix->M; ++i)
	{
		matrix->matrixData[i] = (double*)malloc(matrix->N * sizeof(double)); // Allocate the vector for the columns.
		if (!matrix->matrixData[i])
		{
			printf("Failed to allocate the matrix!\n");
			// Now free the allocated memory so far.
			for (j = i - 1; j >= 0; --i)
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
	int i;
	for (i = 0; i < matrix->M; ++i)
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
	int i, j;
	for (i = 0; i < matrix->M; ++i)
	{
		double sum = i / 1000.0;
		for (j = 0; j < matrix->N; ++j)
		{
			matrix->matrixData[i][j] = j + sum;
		}
	}
}
void matrixSetMinusOnce(struct Matrix* matrix)
{
	if (matrix->matrixData == NULL)
		return; // A little safty..
	int i, j;
	for (i = 0; i < matrix->M; ++i)
	{
		for (j = 0; j < matrix->N; ++j)
		{
			matrix->matrixData[i][j] = -1.0;
		}
	}
}

// Prints the matrix to the standard output.
void matrixPrint(const struct Matrix * matrix)
{
	if (matrix->matrixData == NULL)
		return; // A little safty..
	int i, j;
	for (i = 0; i < matrix->M; ++i)
	{
		for (j = 0; j < matrix->N; ++j)
		{
			printf("%*.3f ", 6, matrix->matrixData[i][j]);
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
			printf("%*.3f ", 6, matrix->matrixData[i][j]);
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
			if (!(tempColumnsData && ((i - 1 < matrix->M % procData->p && i < matrix->M % procData->p) 
									|| (i - 1 >= matrix->M % procData->p && i >= matrix->M % procData->p)))
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
	// The maximum ammount of elements to send is:
	// (NumberOfColumns / p + 1)*(NumberOfRows / p + 1)
	// (every @p-th entry on one row + one extra if not equaly divided between all processes) * (every @p-th row + one extra)
	
	// Note: M is the number of columns and N is the number of rows(the starting matrix is transposed because it's the better way for C)
	int maxEntriesToSend = (procData->M / procData->p + 1) * (procData->N / procData->p + 1);

	// Creates 2D array @p x maxEntriesToSend
	struct Matrix dataToSend;
	matrixAllocate(&dataToSend, procData->p, maxEntriesToSend);
	matrixSetMinusOnce(&dataToSend);
	// Fill the data in the matrix(Note: each row holds the data for correspondig processor, e.g. 2nd row for processor 2..)
	int k, i, j;
	for (k = 0; k < procData->p; ++k)
	{
		double * pRowData = dataToSend.matrixData[k];
		// Each @p-th element on each @p-th row starting from k-th (basicaly each element needed for processor @k)
		
		for (i = k; i < procData->N; i += procData->p)
		{
			// j starts from i-th row and iterates by one column size 
			for (j = i; j < procData->dataCount; j += procData->N) // Note: I keep it transposed, so N is the number of rows(the size of one column)!
			{
				*pRowData = procData->columnsData[j];
				++pRowData;
			}
		}			
	}

	printf("Proc %d has matrix to send:\n", procData->rank);
	matrixPrint(&dataToSend);

	// Send each row of the matrix to the corespondig processor.
	for (k = 0; k < procData->p; ++k)
	{
		if (k != procData->rank)
		{
			// Calculate the data needed to be send to each processor.
			// (Number of rows of this processor which it has to receive) * (Number of columns it has to receive)
			// Not for now, it`s almost not important. 
			MPI_Send(dataToSend.matrixData[k], maxEntriesToSend, MPI_DOUBLE, k, 42, MPI_COMM_WORLD);
		}
	}

	struct Matrix dataToReceive;
	matrixAllocate(&dataToReceive, procData->p, maxEntriesToSend);
	// Send each row of the matrix to the corespondig processor.
	for (k = 0; k < procData->p; ++k)
	{
		if (k == procData->rank)
		{
			for (j = 0; j < maxEntriesToSend; ++j) // simply copy the data.
				dataToReceive.matrixData[k][j] = dataToSend.matrixData[k][j];
		}
		else
		{
			// Calculate the data needed to be send to each processor.
			// (Number of rows of this processor which it has to receive) * (Number of columns it has to receive)
			// Not for now, it`s almost not important. 
			MPI_Recv(dataToReceive.matrixData[k], maxEntriesToSend, MPI_DOUBLE, k, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	}

	printf("Proc %d has matrix received:\n", procData->rank);
	matrixPrint(&dataToReceive);
}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	
	struct ProcData procData;
	struct Matrix matrix; // Only used in processor with rank 0.
	
	
	// I make the number of rows and cols to be in all processors, other way is to broadcast them.
	// TODO Broadcast them!
	int ROWS = 4;
	int COLS = 5;

	// Keep the data transposed.
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
		printf("Matrix:\n");
		matrixPrintTransposed(&matrix);
		printf("End of matrix:\n");
	}
	// Now lets distribute the matrix.(which is stored by columns) // Transposed dimentions COLS <-> ROWS
	distributeColumns(&matrix, &procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	if (procData.rank == 0) // Delete the matrix
	{
		matrixFree(&matrix);
		matrixAllocate(&matrix, COLS, ROWS); // Allocate memory for new matrix
	}

	// Transpose the columns data.
	transpose(&procData);

	//printf("%d has to send %d\n", procData.rank, procData.dataCount);
	// Get the matrix data from all processes.
	selectColumns(&matrix, &procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	if (procData.rank == 0)
	{
		printf("Received matrix:\n");
	//	matrixPrintTransposed(&matrix);
		matrixFree(&matrix);
	}

	MPI_Finalize();
	return 0;
}