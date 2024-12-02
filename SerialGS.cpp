#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
// Function for the Gauss-Seidel algoritm
void ResultCalculation(double* pMatrix, int Size, double &Eps, int &Iterations) {
    double dm, dmax,temp;
    int i, j; // Loop variables
    Iterations = 0;
    do {
        dmax = 0;
        for (i = 1; i < Size - 1; i++)
            for(j = 1; j < Size - 1; j++) {
                temp = pMatrix[Size * i + j];
                pMatrix[Size * i + j] = 0.25 * (pMatrix[Size * i + j + 1] +
                                                pMatrix[Size * i + j - 1] +
                                                pMatrix[Size * (i + 1) + j] +
                                                pMatrix[Size * (i - 1) + j]);
                dm = fabs(pMatrix[Size * i + j] - temp);
                if (dmax < dm) dmax = dm;
            }
        Iterations++;
    }
    while (dmax > Eps);
}

// Function for computational process termination
void ProcessTermination(double* pMatrix) {
    delete [] pMatrix;
}
// Function for formatted matrix output
void PrintMatrix (double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i=0; i<RowCount; i++) {
        for (j=0; j<ColCount; j++)
            printf("%7.4f ", pMatrix[i*ColCount+j]);
        printf("\n");
    }
}

// Function for simple setting the grid node values
void DummyDataInitialization (double* pMatrix, int Size) {
    int i, j; // Loop variables
    double h = 1.0 / (Size - 1);
    // Setting the grid node values
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++)
            if ((i==0) || (i== Size-1) || (j==0) || (j==Size-1))
                pMatrix[i*Size+j] = 100;
            else
                pMatrix[i*Size+j] = 0;
    }
}

// Function for memory allocation and initialization of grid nodes
void ProcessInitialization (double* &pMatrix, int &Size, double &Eps) {
    // Setting the grid size
    // do {
    //     printf("\nEnter the grid size: ");
    //     scanf("%d", &Size);
    //     //printf("\nChosen grid size = %d", Size);
    //     if (Size <= 2)
    //         printf("\nSize of grid must be greater than 2!\n");
    // } while (Size <= 2);
    // Setting the required accuracy
    do {
        //printf("\nEnter the required accuracy: ");
        //scanf("%lf", &Eps);
        //printf("\nChosen accuracy = %lf", Eps);
        Eps = 0.1;  // Setting static accuracy mentioned in the materials (p.21)
        //if (Eps <= 2)
        //    printf("\nAccuracy must be greater than 0!\n");
    } while (Eps <= 0);
    // Memory allocation
    pMatrix = new double [Size*Size];
    // Setting the grid node values
    DummyDataInitialization(pMatrix, Size);
}

int main() {
    double* pMatrix; // Matrix of the grid nodes
    int Size; // Matrix size
    double Eps; // Required accuracy
    int Iterations; // Iteration number

    time_t start, finish;
    double duration;

    // Test cases
    int gridSizes[] = {10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    int numSizes = sizeof(gridSizes) / sizeof(gridSizes[0]); // Number of grid sizes

    printf ("Serial Gauss - Seidel algorithm\n");
    // Process initialization
    for (int i = 0; i < numSizes; i++) {
        int gridSize = gridSizes[i];

        printf("Grid size: %d\n", gridSize);
        ProcessInitialization(pMatrix, Size = gridSize, Eps);
        // Matrix output
        //printf ("Initial Matrix: \n");
        // The Gauss-Seidel method

        start = clock();
        ResultCalculation(pMatrix, Size, Eps, Iterations);
        finish = clock();
        duration = (finish - start) / double(CLOCKS_PER_SEC);

        // Printing the result
        printf("\n Number of iterations: %d\n",Iterations);
        //printf ("\n Result matrix: \n");

        // Printing the execution time of the Gauss-Seidel method
        printf("\n Time of execution: %f\n", duration);

        // Computational process termination
        ProcessTermination(pMatrix);
    }
    return 0;
}