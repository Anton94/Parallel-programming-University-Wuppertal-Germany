#ifndef TWODIMARRAYS_H__
#define TWODIMARRAYS_H__


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
int variable2DArrayAllocate(struct Variable2DArray* variable2Darray);

// Frees the allocated memory for the 2D array. Sets default values.
void variable2DArrayFree(struct Variable2DArray* variable2Darray);

// Prints the matrix to the standard output.
void variable2DArrayPrint(const struct Variable2DArray * variable2Darray);


/* Matrix struct .. basic.. */
struct Matrix
{
	int M, N; // Rows and cols number
	double ** matrixData;
};
// Generates new matrix by the given M and N.
// Need manualy to take care to delete the old matrix.
// Returns 0 if it manage to allocate the memory for the matrix and 0 if not.
int matrixAllocate(struct Matrix* matrix, int m, int n);

// Frees the allocated memory for the matrix. Sets M and N to 0.
void matrixFree(struct Matrix* matrix);

// Sets the the values for the all matrix entries by the given @pFunc function, which for every i and j (row and column indexes) gives double value.
void matrixSetValues(struct Matrix* matrix, double (*pFunc)(int, int));

// Prints the matrix to the standard output.
void matrixPrint(const struct Matrix * matrix);

// Prints the matrix to the standard output.
int matrixCompareWithOtherMatrix(const struct Matrix * matrixOne, const struct Matrix* matrixTwo);

// Prints the matrix to the standard output TRANSPOSED.
void matrixPrintTransposed(const struct Matrix * matrix);


#endif