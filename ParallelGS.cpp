#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <algorithm>

static int ProcNum = 0;   // Number of available processes
static int ProcRank = -1; // Rank of current process

// Function for distribution of the grid rows among the processes
void DataDistribution(double* pMatrix, double* pProcRows, int RowNum, int Size) {
    int *pSendNum;  // Number of elements sent to the process
    int *pSendInd;  // Index of the first data element sent to the process
    int RestRows = Size;

    // Alloc memory for temporary objects
    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];

    // Define the disposition of the matrix rows for current process
    RowNum = (Size - 2) / ProcNum + 2;
    pSendNum[0] = RowNum * Size;
    pSendInd[0] = 0;

    for (int i = 1; i < ProcNum; i++) {
        RestRows = RestRows - RowNum + 1;
        RowNum = (RestRows - 2) / (ProcNum - i) + 2;
        pSendNum[i] = RowNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1] - Size;
    }

    // Scatter the rows
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendInd;
    delete[] pSendNum;
}

// Function for computational process termination
void ProcessTermination(double* pMatrix, double* pProcRows) {
    if (ProcRank == 0)
        delete[] pMatrix;
    delete[] pProcRows;
}

// Function for formatted matrix output
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    for (int i = 0; i < RowCount; i++) {
        for (int j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        printf("\n");
    }
}

// Function for the execution of the Gauss-Seidel method iteration
double IterationCalculation(double* pProcRows, int Size, int RowNum) {
    double dm, dmax, temp;
    dmax = 0;

    for (int i = 1; i < RowNum - 1; i++) {
        for (int j = 1; j < Size - 1; j++) {
            temp = pProcRows[Size * i + j];
            pProcRows[Size * i + j] = 0.25 * (pProcRows[Size * i + j + 1] +
                                              pProcRows[Size * i + j - 1] +
                                              pProcRows[Size * (i + 1) + j] +
                                              pProcRows[Size * (i - 1) + j]);
            dm = fabs(pProcRows[Size * i + j] - temp);
            if (dmax < dm) dmax = dm;
        }
    }
    return dmax;
}

// Function for simple setting the grid node values
void DummyDataInitialization(double* pMatrix, int Size) {
    double h = 1.0 / (Size - 1);

    // Setting the grid node values
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            if ((i == 0) || (i == Size - 1) || (j == 0) || (j == Size - 1))
                pMatrix[i * Size + j] = 100;
            else
                pMatrix[i * Size + j] = 0;
        }
    }
}

// Function for memory allocation and initialization of grid nodes
void ProcessInitialization(double*& pMatrix, double*& pProcRows, int& Size,
                           int& RowNum, double& Eps) {
    int RestRows;

    // Setting the grid size
    if (ProcRank == 0) {
        printf("Grid size: %d\n", Size);
        // do {
        //     printf("\nEnter the grid size: ");
        //     scanf("%d", &Size);
        //     if (Size <= 2) {
        //         printf("\n Size of grid must be greater than 2! \n");
        //     }
        //     if (Size < ProcNum) {
        //         printf("Size of grid must be greater than"
        //                "the number of processes! \n");
        //     }
        // } while ((Size <= 2) || (Size < ProcNum));

        // Setting the required accuracy
        // do {
        //     printf("\nEnter the required accuracy: ");
        //     scanf("%lf", &Eps);
        //     printf("\nChosen accuracy = %lf", Eps);
        //
        //     if (Eps <= 0)
        //         printf("\nAccuracy must be greater than 0!\n");
        // } while (Eps <= 0);
        Eps = 0.1;
    }

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Define the number of matrix rows stored on each process
    RestRows = Size;
    for (int i = 0; i < ProcRank; i++)
        RestRows = RestRows - RestRows / (ProcNum - i);

    RowNum = (RestRows - 2) / (ProcNum - ProcRank) + 2;

    // Memory allocation
    pProcRows = new double[RowNum * Size];

    // Define the values of initial objects’ elements
    if (ProcRank == 0) {
        pMatrix = new double[Size * Size];
        DummyDataInitialization(pMatrix, Size);
    }
}

// Function to copy the initial data
void CopyData(double *pMatrix, int Size, double *pSerialMatrix) {
    std::copy(pMatrix, pMatrix + Size, pSerialMatrix);
}

// Function for exchanging the boundary rows of the process stripes
void ExchangeData(double* pProcRows, int Size, int RowNum) {
    MPI_Status status;

    int NextProcNum = (ProcRank == ProcNum - 1) ? MPI_PROC_NULL : ProcRank + 1;
    int PrevProcNum = (ProcRank == 0) ? MPI_PROC_NULL : ProcRank - 1;

    // Send to NextProcNum and receive from PrevProcNum
    MPI_Sendrecv(pProcRows + Size * (RowNum - 2), Size, MPI_DOUBLE,
                 NextProcNum, 4, pProcRows, Size, MPI_DOUBLE, PrevProcNum, 4,
                 MPI_COMM_WORLD, &status);

    // Send to PrevProcNum and receive from NextProcNum
    MPI_Sendrecv(pProcRows + Size, Size, MPI_DOUBLE, PrevProcNum, 5,
                 pProcRows + (RowNum - 1) * Size, Size, MPI_DOUBLE, NextProcNum, 5,
                 MPI_COMM_WORLD, &status);
}

// Function for the parallel Gauss-Seidel method
void ParallelResultCalculation(double* pProcRows, int Size, int RowNum, double Eps, int& Iterations) {
    double ProcDelta, Delta;
    Iterations = 0;

    do {
        Iterations++;

        // Exchanging the boundary rows of the process stripe
        ExchangeData(pProcRows, Size, RowNum);

        // The Gauss-Seidel method iteration
        ProcDelta = IterationCalculation(pProcRows, Size, RowNum);

        // Calculating the maximum value of the deviation
        MPI_Allreduce(&ProcDelta, &Delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } while (Delta > Eps);
}

// Function for gathering the calculation results
void ResultCollection(double* pMatrix, double* pProcRows, int Size, int RowNum) {
    MPI_Gather(pProcRows + Size, (RowNum - 2) * Size, MPI_DOUBLE, pMatrix + Size,
               (RowNum - 2) * Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Main function
int main(int argc, char* argv[]) {
    double* pMatrix = nullptr;
    double* pProcRows = nullptr;
    int Size;
    int RowNum;
    double Eps;
    int Iterations;
    double start, finish, duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        printf("Parallel Gauss - Seidel algorithm \n");
    }

    // Test cases
    int gridSizes[] = {10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    int numSizes = sizeof(gridSizes) / sizeof(gridSizes[0]); // Number of grid sizes

    for (int i = 0; i < numSizes; i++) {
        int gridSize = gridSizes[i];

        // Process initialization
        ProcessInitialization(pMatrix, pProcRows, Size = gridSize, RowNum, Eps);

        //printf("Process %d: RowNum = %d\n", ProcRank, RowNum);
        MPI_Barrier(MPI_COMM_WORLD);

        // Data distribution among processes
        DataDistribution(pMatrix, pProcRows, RowNum, Size);

        // Start timing
        start = MPI_Wtime();

        // Parallel Gauss-Seidel method
        ParallelResultCalculation(pProcRows, Size, RowNum, Eps, Iterations);

        // Gather results
        ResultCollection(pMatrix, pProcRows, Size, RowNum);

        // End timing
        finish = MPI_Wtime();
        duration = finish - start;

        if (ProcRank == 0) {
            printf("Number of iterations: %d\n", Iterations);
            printf("Execution time: %.6f seconds\n", duration);
        }

        // Free memory
        ProcessTermination(pMatrix, pProcRows);
    }

    MPI_Finalize();
    return 0;
}
