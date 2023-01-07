
#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}


//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}



int main(int argc, char **argv) {

    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);

    printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = (DataType)rand() / (DataType)RAND_MAX;
        hostInput2[i] = (DataType)rand() / (DataType)RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }


  //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
    double start = get_time();
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    double end1 = get_time() - start;
    printf("Copy time CPU to GPU: %f ms \n", end1 * 1000);


  //@@ Initialize the 1D grid and block dimensions here
    int blockDim = 256;
    int gridDim = (inputLength + blockDim - 1) / blockDim;


  //@@ Launch the GPU Kernel here
    start = get_time();
    vecAdd<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    double end2 = get_time() - start;
    printf("Kernel time: %f ms \n", end2 * 1000);



  //@@ Copy the GPU memory back to the CPU here
    start = get_time();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    double end3 = get_time() - start;
    printf("Copy time GPU to CPU: %f ms \n", end3 * 1000);
    printf("Total Execution time: %f ms \n", (end1 + end2 + end3) * 1000);

  //@@ Insert code below to compare the output with the reference
    int comparisonCheck = 1;
    for (int i = 0; i < inputLength; i++) {
        if (hostOutput[i] - resultRef[i] > 1e-6) {
            comparisonCheck = 0;
            break;
        }
    }

    if (comparisonCheck == 1) {
        printf("The comparison is equal!\n");
    } else {
        printf("The comparison is not equal.\n");
    }





  //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

  //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

  return 0;
}
