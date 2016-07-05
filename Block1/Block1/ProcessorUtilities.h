#ifndef PROCESSORUTILITIES_H__
#define PROCESSORUTILITIES_H__

/* Structur to keep the needed data for the processor 
 * and some helping functions for column-cycling wrapping of the matrix entries.
 */


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
	int M, N;// M - Number of columns and N - Number of rows.
	// So in the transposed one (one I use) - M is the number of rows, while each row keeps a column of the original matrix.
};

// Returns tohe total number of entries of the columns of the given processor rank(@rank).
// I don't use the structur ProcData, because it may be called for different processors.
int getTheTotalNumberOfElementsInProcColumns(int rank, int p, int M, int N);

// Allocates vector for the data of all columns of the given processor.
// Column one after that column two and so on...
// Also sets the total number of entries in all columns which are in the given processor.
void allocateOneDimArrayForMultipleColsOfGivenProc(struct ProcData * procData);

// I don't use ProcData structure because it's not necessary to fill own data, but for other processor.
void fillDataOfOneDimColumnsArray(const struct Matrix * matrix, double * data, int rank, int p, int M, int N);

// Deletes the allocated memory for the processor columns.
void freeProcessAllocatedMemory(struct ProcData * procData);

#endif

