#pragma once

namespace DDA {
	template<typename T>
	inline void PackMatrixA(T *A, int blockrows, int m, int block_k, T *packA) {
		for (int j = 0; j < block_k; ++j) {
			T *ptr = A + j * m;
			memcpy(packA + j * blockrows, ptr, sizeof(T)*blockrows);
		}
	}

	template<typename T>
	inline void PackMatrixA_pad(T *A, int padStep, int m, int block_k, T *packA, int padRows) {
		int nonpadrows = padStep - padRows;
		for (int j = 0; j < block_k; ++j) {
			T *ptr = A + j * m;
			T *packA_ptr = packA + j * padStep;
			memcpy(packA_ptr, ptr, sizeof(T)*nonpadrows);
			std::fill_n(packA_ptr + nonpadrows, padRows, T(0));
		}
	}

	template<typename T>
	inline void PackMatrixB(T *B, int blockcols, int block_k, int k, T *packB) {
		for (int j = 0; j < blockcols; ++j) {
			for (int i = 0; i < block_k; ++i) {
				*(packB + i * blockcols + j) = *(B + i + j * k);
			}
		}
	}

	template<typename T>
	inline void PackMatrixB_pad(T *B, int blockcols, int block_k, int k, int pad_cols, T *packB) {
		int i, j, q;
		for (i = 0; i < block_k; ++i) {
			for (j = 0; j < blockcols - pad_cols; ++j) {
				*(packB + i * blockcols + j) = *(B + i + j * k);
			}
			for (q = 0; q < pad_cols; ++q) {
				*(packB + i * blockcols + j + q) = 0;
			}
		}
	}

	template<typename T>
	inline void PackMatrixA_final(T *A, const int& InnerKernel_rows, const int& m, const int& rk, const int& rm, const int& pad_rows, const int& after_pad, const int& padStep, T *packedA) {
		int EndVec = (rm / InnerKernel_rows) * InnerKernel_rows;
		int i, j;
#pragma omp parallel
		{
#pragma omp for schedule(static) private(i,j) nowait
			for (j = 0; j < EndVec; j += InnerKernel_rows) {
				PackMatrixA(A + j, InnerKernel_rows, m, rk, packedA + rk * j);
			}
			for (i = EndVec; i < rm; i += padStep) {
				int now_pad_rows = i + padStep - rm > 0 ? i + padStep - rm : 0;
				PackMatrixA_pad(A + i, padStep, m, rk, packedA + i * rk, now_pad_rows);
			}
		}
	}

	template<typename T>
	inline void PackMatrixB_final(T *B, const int& InnerKernel_cols, const int& n, const int& rk, const int& k, const int& pad_cols, T *packedB) {
		int EndVec = n / InnerKernel_cols * InnerKernel_cols;
#pragma omp parallel //shared(B,packedB)
		{
#pragma omp for schedule(static) nowait
			for (int j = 0; j < EndVec; j += InnerKernel_cols) {
				PackMatrixB(B + j * k, InnerKernel_cols, rk, k, packedB + j * rk);
			}
#pragma omp for schedule(static) nowait
			for (int j = EndVec; j < n; j += InnerKernel_cols) {
				PackMatrixB_pad(B + j * k, InnerKernel_cols, rk, k, pad_cols, packedB + j * rk);
			}
		}
	}

	template<typename T>
	inline void PackMatrix_T(T *mat, const int& kernel_rows, const int& kernel_cols, const int& rows, const int& cols, T *buffer, const int& buffer_size) {
//#pragma omp parallel
		{
//#pragma omp for schedule(static) nowait
			for (int i = 0; i < rows; i += kernel_rows) {
				int real_pack_rows = rows - i > kernel_rows ? kernel_rows : rows - i;
				for (int j = 0; j < kernel_cols; ++j) {
					memcpy(buffer + j * kernel_rows + i * kernel_cols, mat + j * rows + i, sizeof(T)*real_pack_rows);
				}
			}
		}
	}
}