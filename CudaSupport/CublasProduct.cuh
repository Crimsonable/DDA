
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

extern "C" void cudaProductS(float* A, float* B, float* C, int m, int n, int k);
extern "C" void cudaProductD(double* A, double* B, double* C, int m, int n, int k);

