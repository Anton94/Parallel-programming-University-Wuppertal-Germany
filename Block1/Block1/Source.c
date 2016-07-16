#include <stdlib.h>
#include "mpi.h"
#include "TwoDimArrays.h"
#include "MatrixDTS.h"
#include "GaussianElimination.h"

// MatrixDTS -> matrix distribute transpose select.

// Testing the functionality of MatrixDTS for multiple matrixes with row sizes from [a, b] and column sizes from [c, d]
void testMatrixDTS(int a, int b, int c, int d, int outputingTestStatus, int outputingMatrixValues)
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
			res = functionalityDTS(&procData, i, j, outputingMatrixValues, &tSum);
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


// Testing the functionality of MatrixDTS for multiple A square matrixes with sizes [a, b] and B matrixes (NxQ) with column sizes from [c, d].
void testMatrixGE(int a, int b, int c, int d, int outputingTestStatus, int outputingMatrixValues)
{
	struct ProcData procData;

	MPI_Comm_rank(MPI_COMM_WORLD, &procData.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procData.p);

	double tSum = 0.0;

	int i, j;
	int res;
	int failedTests = 0, totalTests = (b - a + 1) * (d - c + 1); // Because it`s closed interval.
	for (i = a; i <= b; ++i) // Size of matrix A (IxI)
	{
		for (j = c; j <= d; ++j) // Number of columns of matrix B(IxJ)
		{
			res = functionalityGE(&procData, i, j, outputingMatrixValues, &tSum);
			if (procData.rank == 0)
			{
				if (!res)
					++failedTests;
				if (outputingTestStatus)
				{
					printf("Testing with matrix A %d-by-%d and matrix B %d-by-%d...\n", i, i, i, j);
				/*	if (res)
						printf("\t ...success!\n");
					else
						printf("\t ...failed!\n");*/
				}
			}
		}
	}

	if (procData.rank == 0)
	{
		printf("Testing with matrix with square matrixes A with sizes from [%d, %d] and matrixes B with columns from [%d, %d] \n\t...took %.9f seconds!\n", a, b, c, d, tSum);
	//	printf("\t...with %d tests failed out of %d!\n", failedTests, totalTests);
	}
}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

//	testMatrixDTS(1, 300, 1, 300, 0, 0);
	testMatrixDTS(5000, 5001, 5000, 5002, 0, 0);
//	testMatrixDTS(200, 290, 200, 290, 1, 0);
//	testMatrixDTS(11, 11, 9, 10, 1, 1);
//	testMatrixDTS(5, 5, 2, 2, 1, 1);
//	testMatrixDTS(11, 11, 9, 10, 1, 1);

//	testMatrixGE(1, 150, 1, 150, 0, 0);
//	testMatrixGE(220, 240, 220, 240, 1, 0);
//	testMatrixGE(5, 5, 2, 2, 1, 1);

	MPI_Finalize();

	return 0;
}