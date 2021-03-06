#pragma once
#include "forwardDecleration.h"
#include "GemmKernels.h"

namespace CSM {
	//using namespace CSM::SSE_OP;
	/*template<typename T>
	class ProductHanlder {
	private:
		static constexpr int InnerKernelRows = 16 / (sizeof(T) / sizeof(float));
		static constexpr int InnerKernelCols = 4;
		//static constexpr int InnerKernelCols = 8;
		static constexpr int mainStep = InnerKernelRows;
		int padStep = 0, leftStep, EndVec, padRows = 0, rm, rk, rn, packedARows, packedBCols, lda, n, ldb;
		T *packedA_ptr, *packedB_ptr, *C_ptr;
	public:
		ProductHanlder() {}
		ProductHanlder(const int& lda, const int& n, const int& ldb, int rm, int rk, int rn, int packedARows, int packedBCols, T *packedA_ptr, T *packedB_ptr, T *C_ptr) {
			this->lda = lda;
			this->n = n;
			this->ldb = ldb;
			this->rm = rm;
			this->rk = rk;
			this->rn = rn;
			this->packedARows = packedARows;
			this->packedBCols = packedBCols;
			this->packedA_ptr = packedA_ptr;
			this->packedB_ptr = packedB_ptr;
			this->C_ptr = C_ptr;
			padRows = rm % InnerKernelRows > 0 ? InnerKernelRows - rm % InnerKernelRows : 0;
			padStep = InnerKernelRows;
			EndVec = padRows + rm;
		}

		FORCE_INLINE void update(int rm, int rn, int rk, int packedBCols, T *C_ptr) {
			this->C_ptr = C_ptr;
			if (rm != this->rm || rk != this->rk || rn != this->rn) {
				this->rm = rm;
				this->rn = rn;
				this->rk = rk;
				this->packedBCols = packedBCols;
				padRows = rm % InnerKernelRows > 0 ? InnerKernelRows - rm % InnerKernelRows : 0;
				padStep = InnerKernelRows;
				EndVec = padRows + rm;
			}
		}

		inline int GetInnerRows() {
			return InnerKernelRows;
		}

		inline int GetInnerCols() {
			return InnerKernelCols;
		}

		inline int GetPadRows() {
			return padRows;
		}

		inline int GetPadStep() {
			return padStep;
		}

		inline int GetTotalRows() {
			return padRows + rm;
		}

		inline void GEMM_InnerLoop() {
#pragma omp parallel
			{
#pragma omp for schedule(dynamic) nowait
				for (int j = 0; j < packedBCols; j += InnerKernelCols) {
					for (int i = 0; i < EndVec; i += InnerKernelRows) {
						AddDot8x4<T, v_256<T>>(packedA_ptr + i * rk, packedB_ptr + j * rk, C_ptr + i + j * lda, rk, n, lda, InnerKernelRows, 1, j, padRows,i + InnerKernelRows > rm);
					}
				}
			}
		}

		inline void GEMP_InnerLoop() {
#pragma omp parallel
			{
#pragma omp for schedule(dynamic) nowait
				for (int rowIndex = 0; rowIndex < EndVec; rowIndex += InnerKernelRows) {
					for (int colIndex = 0; colIndex < packedBCols; colIndex += 1) {
						AddDot8x1<T, v_256<T>>(packedA_ptr + rowIndex * rk, packedB_ptr + colIndex * rk, C_ptr + rowIndex + colIndex * lda, rk, lda, n, ldb, InnerKernelRows, 1, colIndex, padRows, rowIndex + InnerKernelRows > rm);
					}
				}
			}

		}
	};*/

	template<typename T>
	inline void GemmInnerLoop(T* a, const int& lda, T* b, const int& ldb, T* c, const int& ldc, int m, int n, int k, const int& kernelRows,const int& kernelCols) {
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) nowait
			for (int colIndex = 0; colIndex < n; colIndex += kernelCols) {
				int offset_b = colIndex * k;
				int offset_c = colIndex * ldc;
				int leftCols = n - colIndex > 4 ? 4 : n - colIndex;
				for (int rowIndex = 0; rowIndex < m; rowIndex += kernelRows) {
					Gemm_kernel_avx256(a + rowIndex * k, lda, b + offset_b, 1, c + rowIndex + offset_c, ldc, k, m - rowIndex > 16 ? 16 : m - rowIndex, leftCols);
				}
			}
		}
	}
}
