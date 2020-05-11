#pragma once
#include "GemmKernels.h"
#include "Pack.h"
#include "memory.h"
#include "PackImp.h"
using namespace GemmCpuKernels;

#define RowsDevide 110
#define ColsDevide 110
//#define KDevide 256
#define Kernel 8
#define _min_(i, j) ((i) < (j) ? (i) : (j))

namespace DDA {
	template<typename T>
	void PackLhs(T *lhs, int ld, int rows, int cols, T *pack, int ld2, int kernelrows, int kernelcols) {
		const int col_block_count = cols % kernelcols ? cols / kernelcols + 1 : cols / kernelcols;
#pragma omp parallel
		{
			for (int rowIndex = 0; rowIndex < rows; rowIndex += kernelrows) {
#pragma omp for nowait
				for (int colIndex = 0; colIndex < cols; colIndex += kernelcols) {
					int offset = colIndex / 8 * 64 + rowIndex / 8 * col_block_count * 64;
					pack_8x8(lhs + rowIndex + colIndex * ld, ld, (rows - rowIndex) > 8 ? 8 : (rows - rowIndex), (cols - colIndex) > 8 ? 8 : (cols - colIndex), pack + offset, ld2);
				}
			}
		}
	}

	template<typename T>
	void PackRhs(T *rhs, int ld, int rows, int cols, T *pack, int ld2, int kernelrows, int kernelcols) {
		const int row_block_count = rows % kernelrows ? rows / kernelrows + 1 : rows / kernelrows;
#pragma omp parallel
		{
#pragma omp for nowait
			for (int colIndex = 0; colIndex < cols; colIndex += kernelcols) {
				for (int rowIndex = 0; rowIndex < rows; rowIndex += kernelrows) {
					int offset = rowIndex / 8 * 64 + colIndex / 8 * row_block_count * 64;
					pack_8x8_transpose(rhs + rowIndex + colIndex * ld, ld, (rows - rowIndex) > 8 ? 8 : (rows - rowIndex), (cols - colIndex) > 8 ? 8 : (cols - colIndex), pack+offset, ld2);
				}
			}
		}
	}


	template<typename T>
	struct GemmImp {
		T *packA = nullptr, *packB = nullptr;
		~GemmImp() {
			if (packA) aligned_free(packA);
			if (packB) aligned_free(packB);
		}

		inline void GemmInnerDispatch(T* a, int lda, T* b, int ldb, T* c, int ldc, int rows, int cols, int k, bool start) {
			const int packARows = rows % Kernel ? (rows + Kernel - rows % Kernel) : rows;
			const int packBCols = cols % Kernel ? (cols + Kernel - cols % Kernel) : cols;
			const int packACols = k % Kernel ? (k + Kernel - k % Kernel) : k;
			const int kblock_count8x8 = packACols / 8;

			if (!packA && !packB) {
				packA = mynew<T>(packACols*packARows, VECTORIZATION_ALIGN_BYTES);
				packB = mynew<T>(packACols*packBCols, VECTORIZATION_ALIGN_BYTES);
			}
			PackLhs(a, lda, rows, k, packA, 8, 8, 8);
			if (start)
				PackRhs(b, ldb, k, cols, packB, 8, 8, 8);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(packA, 8, packACols*packARows / 8, "packA:");
			DEBUG_TOOLS::printRawMatrix(packB, 8, packACols*packBCols / 8, "packedB:");
			DEBUG_TOOLS::printRawMatrix(a, rows, k, "A:");
			DEBUG_TOOLS::printRawMatrix(b, k, cols, "B:");
#endif // DEBUG_INFO

#pragma omp parallel
			{
				for (int colIndex = 0; colIndex < cols; colIndex += 8) {
					for (int rowIndex = 0; rowIndex < rows; rowIndex += 8) {
#pragma omp for schedule(static) nowait
						for (int kIndex = 0; kIndex < k; kIndex += 8) {
							int offset_a = kIndex / 8 * 64 + rowIndex / 8 * kblock_count8x8 * 64;
							int offset_b = kIndex / 8 * 64 + colIndex / 8 * kblock_count8x8 * 64;
							kernel_256_avx(packA + offset_a, 8, packB + offset_b, 1, c + ldc * colIndex + rowIndex, ldc, rows - rowIndex > 8 ? 8 : (rows - rowIndex), cols - colIndex > 8 ? 8 : (cols - colIndex));
						}
					}
				}
			}
		}
	};

	template<typename T>
	void Gemm(T* a, int lda, T* b, int ldb, T* c, int ldc, int m, int n, int k) {
		int C_RowsStep = RowsDevide, C_ColsStep = ColsDevide;
		GemmImp<T> handle;
		for (int colIndex = 0; colIndex < n; colIndex += C_ColsStep) {
			C_ColsStep = _min_(C_ColsStep, n - colIndex);
			for (int rowIndex = 0; rowIndex < m; rowIndex += C_RowsStep) {
				C_RowsStep = _min_(C_RowsStep, m - rowIndex);
				handle.GemmInnerDispatch(a + rowIndex, lda, b + colIndex * ldb, ldb, c + rowIndex + colIndex * ldc, ldc, C_RowsStep, C_ColsStep, k, rowIndex==0);
			}
		}
	}
}