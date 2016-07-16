#include <stdlib.h>
#include "mpi.h"
#include "MatrixDTS.h"

// Returns the entry value of the matrix on row @i and column j.
// The default values for the matrix Aij = j + i / 1000 (transposed...).
double getDTSEntryValue(struct Matrix * matrix, int i, int j)
{
	if (i > matrix->M || j > matrix->N || i < 0 || j < 0)
		return -1.1;
	return (double)i / 1000.0 + (double)j;
}

// Returns -1 for every entry value.
double getMinusOne(int i, int j)
{
	return -1.0;
}

// Distributes columns of MxN matrix over the processors (processor j holds column i if j === i mod p)
// @procData->M and procData->N:
// @M is the number of cols and @N is the number of rows. 
// NOTE: In the ORIGINAL matrix(which is transposed of this one @matrix).
//		 SO, @M in the matrix struture is the number of rows of @matrix, but it's transposed of the one
//		 that I need to distribute, so @M is the number of columns in the matrix that I need to distribute
//		 (@N analogously)
//		 All of this is because of the fact that C is row-wise 2D array, and I need to distribute the 
//		 matrix column-wise!
void distributeColumns(const struct Matrix* matrix, struct ProcData * procData)
{
	// Each processor has to allocate memory for it`s columns.
	// I will store them in one array - first column after that second and so on.
	// Another approach is like 2D array, but we will see which one is better. 
	// (iteration is not so good in the 2D array)
	// Better send/receive of one dim array than 2D one.
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



// Distribute the data using MPI_scatterv. Drowback - need the matrix to be in one array.
void distributeColumnsWithScatterV(const struct Matrix* matrix, struct ProcData * procData)
{
	// Each processor has to allocate memory for it`s columns.
	// I will store them in one array - first column after that second and so on.
	// Another approach is like 2D array, but we will see which one is better. 
	// (iteration is not so good in the 2D array)
	// Better send/receive of one dim array than 2D one.
	allocateOneDimArrayForMultipleColsOfGivenProc(procData);

	double * dataToSend = NULL; // A vector with the data that needs to be send.
	int * sendCounts = NULL; // The number of data for each processor.
	int * displacements = NULL; // The starting positions in the vector for the data to each processor.

	//sendCounts = (int*)malloc(procData->p * sizeof(int)); // The number of data to send to each processor.(index == processor rank)
	//displacements = (int*)malloc(procData->p * sizeof(int)); // The starting positions in the vector for the data to each processor.

	// I chose the 0 processor to distribute the matrix
	if (procData->rank == 0)
	{
		int totalNumberOfEntries = procData->M * procData->N;
		dataToSend = (double*)malloc(totalNumberOfEntries * sizeof(double));
		sendCounts = (int*)malloc(procData->p * sizeof(int)); // The number of data to send to each processor.(index == processor rank)
		displacements = (int*)malloc(procData->p * sizeof(int)); // The starting positions in the vector for the data to each processor.

		double * pDataToSend = dataToSend,
			*pNextDataToSend;

		// Write the data of the matrix in the vector, first the columns of processor 0, after that the columns for processor 1 and so on.
		int i = 0;
		double * pEndDataToSend = pDataToSend + totalNumberOfEntries;
		while (pDataToSend < pEndDataToSend)
		{
			displacements[i] = pDataToSend - dataToSend; // Displacement from the beginning of the vector.

			// Fill the data for processor with rank @i.
			pNextDataToSend = fillDataOfOneDimColumnsArray(matrix, pDataToSend, i, procData->p, procData->M, procData->N);			
			sendCounts[i] = pNextDataToSend - pDataToSend; // Count of elements			
			pDataToSend = pNextDataToSend;

			++i;
		}

		// Fill the left processors counts and dipspl with 0's.
		while (i < procData->p)
		{
			sendCounts[i] = displacements[i] = 0;
			++i;
		}
	}

	MPI_Scatterv(dataToSend, sendCounts, displacements, MPI_DOUBLE, procData->columnsData, procData->dataCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(sendCounts);
	free(displacements);
	free(dataToSend);
}



// Collects the columns of all processors and processor 0 writes it to the given matrix.
void selectColumnsWithGatherV(const struct Matrix* matrix, struct ProcData * procData)
{
	double * dataToRecv = NULL; // A vector with the data that needs to be received from all processors.
	int * recvCounts = NULL; // The number of data to be received from each processor.(index == processor rank)
	int * displacements = NULL; // The starting positions in the vector for the data to each processor.

	// Calculate the displacements and counts of elements from each processor.
	if (procData->rank == 0)
	{
		int totalNumberOfEntries = procData->M * procData->N;
		dataToRecv = (double*)malloc(totalNumberOfEntries * sizeof(double));
		recvCounts = (int*)malloc(procData->p * sizeof(int)); // The number of data to send to each processor.(index == processor rank)
		displacements = (int*)malloc(procData->p * sizeof(int)); // The starting positions in the vector for the data to each processor.

		int i;
		double * pDataToRecv = dataToRecv;
		for (i = 0; i < procData->p; ++i)
		{
			recvCounts[i] = getTheTotalNumberOfElementsInProcColumns(i, procData->p, procData->M, procData->N);
			displacements[i] = pDataToRecv - dataToRecv;
			pDataToRecv += recvCounts[i];
		}

	}

	MPI_Gatherv(procData->columnsData, procData->dataCount, MPI_DOUBLE, dataToRecv, recvCounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// Write the received data to the matrix.
	if (procData->rank == 0)
	{
		double * pDataToRecv = dataToRecv;
		for (int i = 0; i < procData->p; ++i)
		{
			int k, j;
			for (k = i; k < matrix->M; k += procData->p) // Write the data to all k rows where k belongs to i-th processor
			{
				for (j = 0; j < matrix->N; ++j) // Write all data.
				{
					matrix->matrixData[k][j] = *pDataToRecv;
					++pDataToRecv;
				}
			}
		}
	}

	free(recvCounts);
	free(displacements);
	free(dataToRecv);
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
		// Write own data to the matrix. TODO: do-while loop and remove the 'if'
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
	for (i = 0; i < procData->p; ++i)
	{
		for (j = i; j < procData->p; ++j)
		{
			if (i != j)
				if (procData->rank == i)
				{
				MPI_Send(dataToSend.arrayData[j], dataToSend.rowSizes[j], MPI_DOUBLE, j, 42, MPI_COMM_WORLD);
				MPI_Recv(dataToReceive.arrayData[j], dataToReceive.rowSizes[j], MPI_DOUBLE, j, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
				}
				else if (procData->rank == j)
				{
					MPI_Recv(dataToReceive.arrayData[i], dataToReceive.rowSizes[i], MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
					MPI_Send(dataToSend.arrayData[i], dataToSend.rowSizes[i], MPI_DOUBLE, i, 42, MPI_COMM_WORLD);
				}
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


// The communication between the processors is a little like merge sort.
// Left side of processors sends and right side recvs, and after that the opposite. (each from left side to every one to the right side)
// Divide the range of processors by 2 and call the function for the two halfs.
// ~ 3 * p  communications in total
// The interval is [l, r)
void transpoeBinaryCommExchange(struct ProcData* procData, struct Variable2DArray* dataToSend, struct Variable2DArray* dataToReceive, int l, int r)
{
	// For interval with less or one processor returns. Or the processors which don't do any work here.
	if (l >= r || r - l <= 1 || procData->rank < l || procData->rank >= r)
		return;

	int extra = -1; // Divide the range by 2, if the range is ODD -> keep the last one separate and after that make the comm with him.
	if ((r - l) % 2 == 1)
	{
		extra = r - 1;
		--r;
	}

	int halfSize = (r - l) / 2;
	int mid = l + halfSize;
	int i, rankInLocalInterval, partner;

	// Left side.
	if (procData->rank >= l && procData->rank < mid)
	{
		rankInLocalInterval = procData->rank - l; // The index [0, ... , halfSize)
		for (i = 0; i < halfSize; ++i)
		{
			// My partner when @i == 0 is with same @rankInLocalInterval as me.
			partner = mid + (rankInLocalInterval + i) % halfSize;
			MPI_Send(dataToSend->arrayData[partner], dataToSend->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD);
			MPI_Recv(dataToReceive->arrayData[partner], dataToReceive->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		}
	}
	// Right side.
	else if (procData->rank >= mid && procData->rank < r) // It's only this processors ranks left..
	{
		rankInLocalInterval = procData->rank - mid; // The index [0, ... , halfSize)
		for (i = 0; i < halfSize; ++i)
		{
			// My partner when @i == 0 is with same @rankInLocalInterval as me.
			partner = l + (halfSize + rankInLocalInterval - i) % halfSize;
			MPI_Recv(dataToReceive->arrayData[partner], dataToReceive->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			MPI_Send(dataToSend->arrayData[partner], dataToSend->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD);
		}
	}

	if (extra >= 0)
	{// exchange info with [l, l+1, ... , r - 1] processors with processor with rank r (the given parameter @r is decremented by 1 so originaly with last processor in the given interval -> r-1)
		if (procData->rank == r)
		{
			for (i = l; i < r; ++i)
			{
				MPI_Send(dataToSend->arrayData[i], dataToSend->rowSizes[i], MPI_DOUBLE, i, 42, MPI_COMM_WORLD);
				MPI_Recv(dataToReceive->arrayData[i], dataToReceive->rowSizes[i], MPI_DOUBLE, i, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			}
		}
		else
		{
			MPI_Recv(dataToReceive->arrayData[r], dataToReceive->rowSizes[r], MPI_DOUBLE, r, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			MPI_Send(dataToSend->arrayData[r], dataToSend->rowSizes[r], MPI_DOUBLE, r, 42, MPI_COMM_WORLD);
		}
	}

	transpoeBinaryCommExchange(procData, dataToSend, dataToReceive, l, mid); // Exchange info between Left half processors of the given interval
	transpoeBinaryCommExchange(procData, dataToSend, dataToReceive, mid, r); // And the right half. NOTE: if there was extra one, it will exchange with all of them and no need to exchange again...
}


// The communication between the processors is a little like merge sort.
// Left side of processors sends and right side recvs, and after that the opposite. (each from left side to every one to the right side)
// Divide the range of processors by 2 and call the function for the two halfs.
// If the number of processors are ODD, make left side with one more processor than the right side
// and in the right side add Ghost processor, with him noone communicates.
// So in each level of binary tree structure of the recursion there will be an extra communication(PARALLEL), which is in total ~logP communications.
// ~ 2 * p + logP communications in total.
// The interval is [l, r)
void transpoeBinaryCommExchangeWithGhostProcessor(struct ProcData* procData, struct Variable2DArray* dataToSend, struct Variable2DArray* dataToReceive, int l, int r)
{
	// For interval with less or one processor returns. Or the processors which don't do any work here.
	if (l >= r || r - l <= 1 || procData->rank < l || procData->rank >= r)
		return;

	// Divide the range by 2, if the range is ODD -> make a ghost processor, noone communicates with him but the communications become easyear.
	int ghost = -1;
	if ((r - l) % 2 == 1)
	{
		ghost = r;
		++r;
	}

	int halfSize = (r - l) / 2;
	int mid = l + halfSize;
	int i, rankInLocalInterval, partner;

	// Left side.
	if (procData->rank >= l && procData->rank < mid)
	{
		rankInLocalInterval = procData->rank - l; // The index [0, ... , halfSize)
		for (i = 0; i < halfSize; ++i)
		{
			// My partner when @i == 0 is with same @rankInLocalInterval as me.
			partner = mid + (rankInLocalInterval + i) % halfSize;

			if (partner != ghost)
			{
				//printf("In stage %d - rocessord %d tries to exchange with processor %d\n", i, procData->rank, partner);
				MPI_Send(dataToSend->arrayData[partner], dataToSend->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD);
				MPI_Recv(dataToReceive->arrayData[partner], dataToReceive->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			}
		}
	}
	// Right side.
	else if (procData->rank >= mid && procData->rank < r) // It's only this processors ranks left..
	{
		rankInLocalInterval = procData->rank - mid; // The index [0, ... , halfSize)
		for (i = 0; i < halfSize; ++i)
		{
			// My partner when @i == 0 is with same @rankInLocalInterval as me.
			partner = l + (halfSize + rankInLocalInterval - i) % halfSize;

			if (partner != ghost)
			{
				//printf("In stage %d - rocessord %d tries to exchange with processor %d\n", i, procData->rank, partner);
				MPI_Recv(dataToReceive->arrayData[partner], dataToReceive->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
				MPI_Send(dataToSend->arrayData[partner], dataToSend->rowSizes[partner], MPI_DOUBLE, partner, 42, MPI_COMM_WORLD);
			}
		}
	}

	if (ghost > -1)
		--r; // execute the transposing for the left side and th right side but without the ghoust processor.

	transpoeBinaryCommExchangeWithGhostProcessor(procData, dataToSend, dataToReceive, l, mid); // Exchange info between Left half processors of the given interval
	transpoeBinaryCommExchangeWithGhostProcessor(procData, dataToSend, dataToReceive, mid, r); // And the right half. NOTE: if there was extra one, it will exchange with all of them and no need to exchange again...
}

// Transpose the matrix over each processor.
void transposeBinaryComm(struct ProcData* procData)
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

	transpoeBinaryCommExchangeWithGhostProcessor(procData, &dataToSend, &dataToReceive, 0, procData->p);

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



// Transpose the matrix over each processor.
// Send the data to 'i-th' next processor and receive it from 'i-th' previous.
// Cyclically somehow..... 
// NOTE: If the send and receive communication is not in one, it wont work...
void transposeCyclicly(struct ProcData* procData)
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

	int partnerSendTo, partnerReceiveFrom;
	for (i = 1; i < procData->p; ++i)
	{
		// Send to next @i-th and receive from prev @i-th
		partnerSendTo = (procData->rank + i) % procData->p;
		partnerReceiveFrom = (procData->p + procData->rank - i) % procData->p; // p + .. because of module operation, not to began negative...
		MPI_Sendrecv(dataToSend.arrayData[partnerSendTo], dataToSend.rowSizes[partnerSendTo], MPI_DOUBLE, partnerSendTo, 42, dataToReceive.arrayData[partnerReceiveFrom], dataToReceive.rowSizes[partnerReceiveFrom], MPI_DOUBLE, partnerReceiveFrom, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
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

/// Main functionality.
// runs the functionality for matrix with @ROWS number of rows, @COLS number of columns and
// if outputing is TRUE it prints the matrixes and 
// adds the time for distribute, transpose and select to @tSum EXEPT the time for creating and freeing the matrixes. (only inportant for @rank = 0
int functionalityDTS(struct ProcData * procData, int ROWS, int COLS, int outputing, double * tSum)
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
	procData->M = dims[1]; // Number of columns of original matrix.
	procData->N = dims[0]; // Number of rows of original matrix.

	/* Create the matrixes in proc 0 */
	// It`s C , so I will keep the matrix 'transposed', because I want to send whole columns,
	// So I will keep the matrix as rows of columns(first row is the first column, second row is the second column and so on).
	if (procData->rank == 0)
	{
		// Generate matrixes
		matrixAllocate(&matrixToSend, procData->M, procData->N);	 // NOTE: here I give transposed matrix, so swaped number of rows and cols.
		matrixAllocate(&matrixToReceive, procData->N, procData->M);// Transposed one but the returned is also transposed

		matrixSetValues(&matrixToSend, &getDTSEntryValue);

		if (outputing)
		{
			printf("Matrix:\n");
			matrixPrintTransposed(&matrixToSend);
		}
	}

	double t1, t2;

	t1 = MPI_Wtime();

	// Now lets distribute the matrix.(which is stored by columns) // Transposed dimentions COLS <-> ROWS
	distributeColumnsWithScatterV(&matrixToSend, procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	// Transpose the columns data.
	transposeBinaryComm(procData);
	MPI_Barrier(MPI_COMM_WORLD); // synchonization.

	// Get the matrix data from all processes.
	selectColumnsWithGatherV(&matrixToReceive, procData);
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