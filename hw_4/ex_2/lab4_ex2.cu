#include <stdio.h>
#include <random>
#include <cuda_runtime.h>

#define DataType double
#define TPB 256

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}


int main(int argc, char **argv) {

    int inputLength;
    int S_seg;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    S_seg = atoi(argv[2]);

    printf("The input length is %d\n", inputLength);
    printf("The segment size is %d\n", S_seg);

    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }

    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost((void **)&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost((void **)&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost((void **)&hostOutput, inputLength * sizeof(DataType));
    cudaMallocHost((void **)&resultRef, inputLength * sizeof(DataType));

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

    int gridDim = (S_seg/ TPB);

    int iterations = inputLength / S_seg + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++){
        int stream = i % 4;
        int offset = i * S_seg;
        int length = min(S_seg, inputLength - offset);
        //printf("The length is %d\n", length);
        //printf("S_seg is %d\n", S_seg);
        int bytes = min(S_seg, inputLength - offset) * sizeof(DataType);

        cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, bytes, cudaMemcpyHostToDevice, streams[stream]);
        cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, bytes, cudaMemcpyHostToDevice, streams[stream]);
        vecAdd<<<gridDim, TPB, 0, streams[stream]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, length);
        cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, bytes, cudaMemcpyDeviceToHost, streams[stream]);
    }
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Execution time is %f ms\n", milliseconds);

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

    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    cudaFree(hostInput1);
    cudaFree(hostInput2);
    cudaFree(hostOutput);
    cudaFree(resultRef);

    return 0;
}
