/*
 * gpu-add.cu
 * Copyright (C) 2022 jht <jht@jhtMac.local>
 *
 * Distributed under terms of the MIT license.
 */

#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>

const int BLOCKSIZE = 64;

int parse_arg(int argc, char* argv[]) {
    int opt;
    int n = -1;
    const char* ERR_MSG = "Usage: %s -n number\n";
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            default:
                fprintf(stderr, ERR_MSG, argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if(n < 0) {
        fprintf(stderr, ERR_MSG, argv[0]);
        exit(EXIT_FAILURE);
    }
    return n;
}

void random_float(float *array, int n) {
    for(int i = 0; i < n; ++i) {
        array[i] = (float)rand();
    }
}

float array_sum(float *array, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i) {
        sum += array[i];
    }
    return sum;
}

__global__ void gpu_add(const float *d_A, const float *d_B, float* d_C, int n) {
    int i = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y +
             threadIdx.x * blockDim.x + threadIdx.y;
    if (i < n) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main(int argc, char* argv[]) {
    int n = parse_arg(argc, argv);
    srand(0);
    timeval start, finish;

    // memory allocation
    int size = n*sizeof(float);
    gettimeofday(&start, 0);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);
    gettimeofday(&finish, 0);
    double t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("CPU Malloc: %lf ms\n", t / 1000);

    gettimeofday(&start, 0);
    float *d_A, *d_B, *d_C;
    cudaError_t error = cudaSuccess;
    error = cudaMalloc((void **)&d_A, size); 
    error = cudaMalloc((void **)&d_B, size); 
    error = cudaMalloc((void **)&d_C, size);
    if (error != cudaSuccess) {
        printf("Fail to cudaMalloc on GPU");
        return EXIT_FAILURE;
    }
    gettimeofday(&finish, 0);
    t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("cudaMalloc: %lf ms\n", t / 1000);
    
    // initialization
    gettimeofday(&start, 0);
    random_float(A, n);
    random_float(B, n);
    gettimeofday(&finish, 0);
    t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("initialize: %lf ms\n", t / 1000);

    gettimeofday(&start, 0);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // calculation
    int gridsize = (int)ceil(sqrt(ceil(n / (BLOCKSIZE * BLOCKSIZE))));
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(gridsize, gridsize, 1);

    gpu_add<<<dimBlock, dimGrid>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    gettimeofday(&finish, 0);
    t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("GPU: %lf ms\n", t / 1000);

    gettimeofday(&start, 0);
    float sum = array_sum(C, n);
    printf("result: %f\n", sum);
    gettimeofday(&finish, 0);
    t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("sum: %lf ms\n", t / 1000);

    // clean up
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
