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
	int M, N;
};


int getTheTotalNumberOfElementsInProcColumns(int rank, int p, int M, int N);

void allocateOneDimArrayForMultipleColsOfGivenProc(struct ProcData * procData);

void fillDataOfOneDimColumnsArray(const struct Matrix * matrix, double * data, int rank, int p, int M, int N);

// Deletes the allocated memory for the processor columns.
void freeProcessAllocatedMemory(struct ProcData * procData);

#endif

