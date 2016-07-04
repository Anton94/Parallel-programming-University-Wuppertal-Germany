#ifndef GAUSSIANELIMINATION_H__
#define GAUSSIANELIMINATION_H__

/* Parallel Gaussian elimination, kji form, of Matrix A - of type NxN, and B - of type NxQ.
 * Ax = B
 *
 *
 * Parallel Triangular  solver. UX = B
 *
 *
 * Notes to myself:
 * In ProcData I keep N as matrix A sizes(square matrix NxN) and M = N + Q where Q is the number of columns of matrix B(which is NxQ)
 */
/*
* The Matrix (A|B) has the memory representation as (N+Q)xN , because each row of the matrix represents one column of the (A|B)
* transposed...
*/


// Returns the entry value of the matrix on row @i and column j.
// The default values for the TRANSPOSED MATRIX (A|B)
// Are, for first N rows(which are first N columns of the original matrix A)
// Aij = (i + j)/n * sin[(i*j*pi)/(n + 1)]
// And for the B part of (A|B) (transposed, so, it's columns are the rows: N+1..M(M=N+Q))
// Bij = i + j
double getGEEntryValue(struct ProcData * procData, int i, int j);

// Distributes columns of MxN matrix over the processors (processor j holds column i if j === i mod p)
// @M is the number of rows and @N is the number of columns.
// Returns pointer to the new allocated memory for the columns...
void distributeColumnsGE(const struct Matrix* matrix, struct ProcData * procData);

// Collects the columns of all processors and processor 0 writes it to the given matrix.
void selectColumnsGE(const struct Matrix* matrix, struct ProcData * procData);

/// Main functionality.
// runs the functionality for matrix with @ROWS number of rows, @COLS number of columns and
// if outputing is TRUE it prints the matrixes and 
// adds the time for distribute, transpose and select to @tSum EXEPT the time for creating and freeing the matrixes. (only inportant for @rank = 0
int functionalityGE(struct ProcData * procData, int ROWS, int COLS, int outputing, double * tSum);

#endif