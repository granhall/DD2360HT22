
#include <stdio.h>
#include <sys/time.h>

#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    DataType sum = 0;
    for (int i = 0; i < numAColumns; i++) {
      sum += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  } else {
    return;
  }



}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
      hostA[i * numAColumns + j] = (DataType)rand() / RAND_MAX;
    }
  }
  for (int i = 0; i < numBRows; i++) {
    for (int j = 0; j < numBColumns; j++) {
      hostB[i * numBColumns + j] = (DataType)rand() / RAND_MAX;
    }
  }

  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      DataType sum = 0;
      for (int k = 0; k < numAColumns; k++) {
        sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
      resultRef[i * numCColumns + j] = sum;
    }
  }



  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  double start = get_time();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double end = get_time() - start;
  printf("Copy time CPU to GPU: %f ms \n", end * 1000);

  //@@ Initialize the grid and block dimensions here
  int blockWidth = 16;
  int blockHeight = 16;
  int gridWidth = (numCColumns + blockWidth - 1) / blockWidth;
  int gridHeight = (numCRows + blockHeight - 1) / blockHeight;


  //@@ Launch the GPU Kernel here
  start = get_time();
  gemm <<<dim3(gridWidth, gridHeight), dim3(blockWidth, blockHeight)>>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  end = get_time() - start;
    printf("Kernel time: %f ms \n", end * 1000);

  //@@ Copy the GPU memory back to the CPU here
  start = get_time();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  end = get_time() - start;
  printf("Copy time GPU to CPU: %f ms \n", end * 1000);

  //@@ Insert code below to compare the output with the reference
  int comparisonCheck = 1;
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      if (hostC[i * numCColumns + j] - resultRef[i * numCColumns + j] > 1e-6) {
        comparisonCheck = 0;
        break;
      }
    }
  }
  if (comparisonCheck) {
    printf("The comparison is equal!\n");
  } else {
    printf("The comparison is not equal.\n");
  }


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
