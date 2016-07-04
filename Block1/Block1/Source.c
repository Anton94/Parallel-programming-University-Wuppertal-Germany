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

//	testMatrixDTS(1, 300, 1, 300, 0, 0);
//	testMatrixDTS(5000, 5001, 5000, 5001, 0, 0);
	//testMatrixDTS(200, 290, 200, 290, 1, 0);
	testMatrixDTS(220, 240, 220, 240, 1, 0);
	//testMatrixDTS(11, 11, 9, 10, 1, 1);

	MPI_Finalize();

	return 0;
}