#include "CublasProduct.cuh"

__global__ static void show(float* C, size_t pitch, int r, int c) {
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			printf("%f ", *((float*)((char*)C + sizeof(float)*i + j * pitch)));
		}
		printf("\n");
	}
}


extern "C" void cudaProductS(float* A, float* B, float* C, int m, int n, int k) {
	float alpha = 1.0, beta = 0.0;
	float* d_A, *d_B, *d_C;
	size_t size_A = sizeof(float)*m*k, size_B = sizeof(float)*k*n, size_C = sizeof(float)*m*n;
	/*std::size_t pitchA, pitchB, pitchC;
	cudaMallocPitch(&d_A, &pitchA, m * sizeof(float), n);
	cudaMallocPitch(&d_B, &pitchB, k * sizeof(float), n);
	cudaMallocPitch(&d_C, &pitchC, m * sizeof(float), n);

	cudaError_t error = cudaMemcpy2D(d_A, pitchA, A, sizeof(float)*m, sizeof(float)*m, k, cudaMemcpyHostToDevice);
	cudaError_t error2 = cudaMemcpy2D(d_B, pitchB, B, sizeof(float)*k, sizeof(float)*k, n, cudaMemcpyHostToDevice);*/
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);

	cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

	/*dim3 grid(1, 1);
	dim3 block(1, 1);
	show << <grid, block >> > (d_A, sizeof(float)*m, m, k);
	show << <grid, block >> > (d_B, sizeof(float)*k, k, n);*/

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
	//show << <grid, block >> > (d_C, sizeof(float)*m, m, n);

	//cudaMemcpy2D(C, m * sizeof(float), d_C, pitchC, sizeof(float)*m, n, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

extern "C" void cudaProductD(double* A, double* B, double* C, int m, int n, int k) {
	double alpha = 1.0, beta = 0.0;
	double* d_A, *d_B, *d_C;
	size_t size_A = sizeof(double)*m*k, size_B = sizeof(double)*k*n, size_C = sizeof(double)*m*n;
	/*std::size_t pitchA, pitchB, pitchC;
	cudaMallocPitch(&d_A, &pitchA, m * sizeof(float), n);
	cudaMallocPitch(&d_B, &pitchB, k * sizeof(float), n);
	cudaMallocPitch(&d_C, &pitchC, m * sizeof(float), n);

	cudaError_t error = cudaMemcpy2D(d_A, pitchA, A, sizeof(float)*m, sizeof(float)*m, k, cudaMemcpyHostToDevice);
	cudaError_t error2 = cudaMemcpy2D(d_B, pitchB, B, sizeof(float)*k, sizeof(float)*k, n, cudaMemcpyHostToDevice);*/
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);

	cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

	/*dim3 grid(1, 1);
	dim3 block(1, 1);
	show << <grid, block >> > (d_A, sizeof(float)*m, m, k);
	show << <grid, block >> > (d_B, sizeof(float)*k, k, n);*/

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
	//show << <grid, block >> > (d_C, sizeof(float)*m, m, n);

	//cudaMemcpy2D(C, m * sizeof(float), d_C, pitchC, sizeof(float)*m, n, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
