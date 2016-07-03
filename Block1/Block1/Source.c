#include <stdlib.h>
#include "mpi.h"
#include "TwoDimArrays.h"
#include "MatrixDTS.h"


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

//	test(100, 120, 100, 120, 0, 0);
//	test(5000, 5001, 5000, 5001, 0, 0);
	//test(200, 290, 200, 290, 1, 0);
	//test(220, 240, 220, 240, 1, 0);
	test(11, 11, 9, 10, 1, 1);

	MPI_Finalize();

	return 0;
}