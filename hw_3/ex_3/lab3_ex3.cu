
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                unsigned int num_elements,
                                unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ unsigned int s_bins[NUM_BINS];

  int binS = blockIdx.x * blockDim.x + threadIdx.x;


  for (int i = threadIdx.x ; i < num_bins; i+= blockDim.x) {
    if(i < num_bins){
      s_bins[i] = 0;
    }
  }

  __syncthreads();

  if (binS < num_elements) {
    atomicAdd(&(s_bins[input[binS]]), 1);
  }
  __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x){
      if(i < num_bins){
        atomicAdd(&bins[i], s_bins[i]);
      }
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  const int binS = blockIdx.x * blockDim.x + threadIdx.x;
//@@ Insert code below to clean up bins that saturate at 127
  if (binS < num_bins) {
    if (bins[binS] > 127) {
      bins[binS] = 127;
    }
  } else {
    return;
  }

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }


  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));

  for (int i = 0; i < inputLength; ++i) {
    resultRef[hostInput[i]]++;
  }

  for (int i = 0; i < NUM_BINS; ++i) {
    if (resultRef[i] > 127) {
      resultRef[i] = 127;
    }
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);


  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  //@@ Initialize the grid and block dimensions here
  int blockSize = 256;
  int gridSize = (inputLength + blockSize - 1) / blockSize;


  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridSize, blockSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);


  //@@ Initialize the second grid and block dimensions here
  int blockSize2 = 256;
  int gridSize2 = (NUM_BINS + blockSize2 - 1) / blockSize2;


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridSize2, blockSize2>>>(deviceBins, NUM_BINS);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();


  //@@ Insert code below to compare the output with the reference
  int comparisonCheck = 1;
  for (int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      comparisonCheck = 0;
      break;
    }
  }
  if (comparisonCheck == 1) {
    printf("The comparison is equal!\n");
  } else {
    printf("The comparison is not equal.\n");
  }

  FILE *path = fopen("data.txt", "w");
  for (int i = 0; i < NUM_BINS; ++i)
    fprintf(path, "%d\n", hostBins[i]);
  fclose(path);

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

