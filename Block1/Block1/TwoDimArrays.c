#include <stdlib.h>
#include <math.h>
#include "TwoDimArrays.h"

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

	free(variable2Darray->arrayData); // Free the array of pointers to the row arrays.
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

// Prints the matrix to the standard output.
int matrixCompareWithOtherMatrix(const struct Matrix * matrixOne, const struct Matrix* matrixTwo)
{
	if (matrixOne->M != matrixTwo->N || matrixOne->N != matrixTwo->M)
		return 0;
	int i, j;
	float epsilon = 0.000000001;
	for (i = 0; i < matrixOne->M; ++i)
	{
		for (j = 0; j < matrixOne->N; ++j)
		{
			if (!abs(matrixOne->matrixData[i][j] - matrixTwo->matrixData[j][i]) < epsilon)
				return 0;
		}
	}

	return 1;
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