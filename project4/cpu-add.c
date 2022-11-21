/*
 * cpu-add.c
 * Copyright (C) 2022 jht <jht@jhtMac.local>
 *
 * Distributed under terms of the MIT license.
 */

#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>

int parse_arg(int argc, char *argv[]) {
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
    if(n == -1) {
        fprintf(stderr, ERR_MSG, argv[0]);
        exit(EXIT_FAILURE);
    }
    return n;
}

void random_float(float* array, int n) {
    for(int i = 0; i < n; ++i) {
        array[i] = (float)rand();
    }
}

void cpu_add(float* A, float* B, float* C, int n) {
    for(int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

float array_sum(float* array, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i) {
        sum += array[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {
    int n = parse_arg(argc, argv);
    srand(0);
    timeval start, finish;

    // memory allocation
    int size = n*sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);
    
    // initialization
    random_float(A, n);
    random_float(B, n);

    // calculation
    gettimeofday(&start, 0);
    cpu_add(A, B, C, n);
    gettimeofday(&finish, 0);
    double t = 1000000 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    printf("CPU: %lf ms\n", t / 1000);
    float sum = array_sum(C, n);
    printf("sum: %f\n", sum);

    free(A); free(B); free(C);
    return 0;
}
