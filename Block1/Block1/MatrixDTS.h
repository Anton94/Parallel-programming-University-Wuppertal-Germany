#ifndef MATRIXDTS_H__
#define MATRIXDTS_H__

#include "TwoDimArrays.h"
#include "ProcessorUtilities.h"

// Matrix - Distribute Transpose Select


// Returns the entry value of the matrix on row @i and column j.
// The default values for the matrix Aij = j + i / 1000 (transposed...).
double getDTSEntryValue(struct Matrix * matrix, int i, int j);

// Returns -1 for every entry value.
double getMinusOne(int i, int j);

// Distributes columns of MxN matrix over the processors (processor j holds column i if j === i mod p)
// @M is the number of cols and @N is the number of rows. 
// NOTE: In the ORIGINAL matrix(which is transposed of this one @matrix).
//		 SO, @M in the matrix struture is the number of rows of @matrix, but it's transposed of the one
//		 that I need to distribute, so @M is the number of columns in the matrix that I need to distribute
//		 (@N analogously)
//		 All of this is because of the fact that C is row-wise 2D array, and I need to distribute the 
//		 matrix column-wise!
void distributeColumns(const struct Matrix* matrix, struct ProcData * procData);

// Distribute the data using MPI_Scatterv.
void distributeColumnsWithScatterV(const struct Matrix* matrix, struct ProcData * procData);

// Collects the columns of all processors and processor 0 writes it to the given matrix.
void selectColumns(const struct Matrix* matrix, struct ProcData * procData);

// Collects the columns of all processors to processor 0 and processor 0 writes it to the given matrix.
// Using MPI_Gatherv routine.
void selectColumnsWithGatherV(const struct Matrix* matrix, struct ProcData * procData);

// Returns the total number of columns I hold.
int transposeColumnsIHold(struct ProcData* procData);

// Returns the ammount of data to be send from @procData->rank processor to @toRank processor.
int transposeGetNumberOfEntriesToSendToProc(struct ProcData* procData, int toRank, int columnsIHold);

// Returns the number of rows given processor needs to receive from transposing.
int transposeRowsToReceive(struct ProcData* procData);

// Returns the ammount of data @procData->rank processor to receive from @toRank processor.
int transposeGetNumberOfEntriesToReceivFromProc(struct ProcData* procData, int fromRank, int rowsToReceive);

// Transpose the matrix over each processor. A lot of communications(p^2)
void transpose(struct ProcData* procData);

// The communication between the processors is a little like merge sort.
// Left side of processors sends and right side recvs, and after that the opposite. (each from left side to every one to the right side)
// Divide the range of processors by 2 and call the function for the two halfs.
// ~ 3 * p  communications in total
// The interval is [l, r)
void transpoeBinaryCommExchange(struct ProcData* procData, struct Variable2DArray* dataToSend, struct Variable2DArray* dataToReceive, int l, int r);

// The communication between the processors is a little like merge sort.
// Left side of processors sends and right side recvs, and after that the opposite. (each from left side to every one to the right side)
// Divide the range of processors by 2 and call the function for the two halfs.
// If the number of processors are ODD, make left side with one more processor than the right side
// and in the right side add Ghost processor, with him noone communicates.
// So in each level of binary tree structure of the recursion there will be an extra communication(PARALLEL), which is in total ~logP communications.
// ~ 2 * p + logP communications in total.
// The interval is [l, r)
void transpoeBinaryCommExchangeWithGhostProcessor(struct ProcData* procData, struct Variable2DArray* dataToSend, struct Variable2DArray* dataToReceive, int l, int r);

// Transpose the matrix over each processor.
void transposeBinaryComm(struct ProcData* procData); 

// Transpose the matrix over each processor.
// Send the data to 'i-th' next processor and receive it from 'i-th' previous.
// Cyclically somehow..... 
// NOTE: If the send and receive communication is not in one, it wont work...
void transposeCyclicly(struct ProcData* procData);

/// Main functionality.
// runs the functionality for matrix with @ROWS number of rows, @COLS number of columns and
// if outputing is TRUE it prints the matrixes and 
// adds the time for distribute, transpose and select to @tSum EXEPT the time for creating and freeing the matrixes. (only inportant for @rank = 0
int functionalityDTS(struct ProcData * procData, int ROWS, int COLS, int outputing, double * tSum);

#endif