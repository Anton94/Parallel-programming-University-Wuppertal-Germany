#ifndef MATRIXDTS_H__
#define MATRIXDTS_H__

// Matrix - Distribute Transpose Select

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

// Returns the entry value of the matrix on row @i and column j.
// The default values for the matrix Aij = j + i / 1000 (transposed...).
double getDTSEntryValue(int i, int j);

// Returns -1 for every entry value.
double getMinusOne(int i, int j);

// Distributes columns of MxN matrix over the processors (processor j holds column i if j === i mod p)
// @M is the number of rows and @N is the number of columns.
// Returns pointer to the new allocated memory for the columns...
void distributeColumnsDTS(const struct Matrix* matrix, struct ProcData * procData);

// Collects the columns of all processors and processor 0 writes it to the given matrix.
void selectColumnsDTS(const struct Matrix* matrix, struct ProcData * procData);

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
// ~ 3/2 * plogp communications in total
// The interval is [l, r)
void transpoeBInaryCommExchange(struct ProcData* procData, struct Variable2DArray* dataToSend, struct Variable2DArray* dataToReceive, int l, int r);

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
int functionality(struct ProcData * procData, int ROWS, int COLS, int outputing, double * tSum);

#endif