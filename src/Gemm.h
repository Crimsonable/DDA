#pragma once
#include "SIMD.h"
#include "forwardDecleration.h"
#include "memory.h"
#include "GemmInnerLoop.h"
#include "Pack.h"

#ifdef HAS_CUDA
#include "CublasProduct.h"
#endif // HAS_CUDA

#ifdef _DEBUG
#include"testTools.h"
#endif // _DEBUG


#define m_kernel 128
#define k_kernel 416
#define inner_rows 16
#define inner_cols 4
#define basic_step 4   
#define _min_(i, j) ((i) < (j) ? (i) : (j))
#define ELEMENT(i, j, k, p) c_##i##j##_c_##k##j##_vreg.d[p]
#define VELEMENT(i,j,k) c_##i##j##_c_##k##j##_vreg.v

/*
A:	num*[m_kernel*k_kernel]		B:	  Bnum*[k_kernel*n]
	 ________  ________				 ____________________
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
	|________||________|			|____________________|
								
*/

namespace CSM {
	
	/*template<typename T>
	class ProductCls {
	private:
		T *packedA = nullptr, *packedB = nullptr;
		ProductHanlder<T> *handler = nullptr;
		static constexpr int KernelRowSize = 16 / sizeof(T) * sizeof(float);

	public:
		~ProductCls() {
			if(packedA) aligned_free(packedA);
			if(packedB) aligned_free(packedB);
			std::free(handler);
		}

		FORCE_INLINE void GEMM_Kernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, const int& lda, int n, const int& ldb, bool isFirst) {
			//alloc memory for packing
			const std::size_t packedARows = rm % KernelRowSize == 0 ? rm : (rm + KernelRowSize - rm % KernelRowSize);
			const std::size_t packedBCols = n % 4 == 0 ? n : (n + 4 - n % 4);

			if (!packedA && !packedB) {
				packedA = mynew<T>(packedARows * rk, VECTORIZATION_ALIGN_BYTES);
				packedB = mynew<T>(packedBCols * rk, VECTORIZATION_ALIGN_BYTES);
			}
			if (!handler) {
				handler = new ProductHanlder<T>(lda, n, ldb, rm, rk, rn, packedARows, packedBCols, packedA, packedB, C);
			}

			handler->update(rm, rn, rk, packedBCols, C);

			//pack InnerKernel A(rm*rk) into packedA
			PackMatrixA_final(b_A, handler->GetInnerRows(), lda, rk, rm, handler->GetPadRows(), handler->GetTotalRows(), handler->GetPadStep(), packedA);
			if (isFirst)
				PackMatrixB_final(b_B, handler->GetInnerCols(), n, rk, ldb, packedBCols - n, packedB);

			handler->GEMM_InnerLoop();
		}

		template<typename T>
		FORCE_INLINE void GEMP_Kernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int lda, int n, int ldb) {
			const std::size_t packedARows = rm % KernelRowSize == 0 ? rm : (rm + KernelRowSize - rm % KernelRowSize);

			if (!packedA) {
				packedA = mynew<T>(packedARows * rk, VECTORIZATION_ALIGN_BYTES);
			}
			if (!handler) {
				handler = new ProductHanlder<T>(lda, n, ldb, rm, rk, rn, packedARows, rn, packedA, b_B, C);
			}

			handler->update(rm, rn, rk, rn, C);

			//pack InnerKernel A(rm*rk) into packedA
			PackMatrixA_final(b_A, handler->GetInnerRows(), lda, rk, rm, handler->GetPadRows(), handler->GetTotalRows(), handler->GetPadStep(), packedA);

			handler->GEMP_InnerLoop();
		}
	};

	template <typename T>
	void Product(T *ptrA, T *ptrB, T *ptrC, const int& lda, const int& ldb, int m, int n, int k) {
#ifndef HAS_CUDA
		int colStep = k_kernel, rowStep = m_kernel;
		ProductCls<T> productInstance;

		if (n > 64) {
			for (int colIndex = 0; colIndex < k; colIndex += colStep) {
				colStep = _min_(k - colIndex, k_kernel);
				for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
					rowStep = _min_(m - rowIndex, m_kernel);
					productInstance.GEMM_Kernel(ptrA + lda * colIndex + rowIndex, ptrB + colIndex, ptrC + rowIndex, rowStep, colStep, n, lda, n, ldb, rowIndex == 0);
				}
			}
		}
		else {
			for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
				rowStep = _min_(m - rowIndex, m_kernel);
				productInstance.GEMP_Kernel(ptrA + rowIndex, ptrB, ptrC + rowIndex, rowStep, k, n, lda, n, ldb);
			}
		}
#else
		cudaProduct(ptrA, ptrB, ptrC, A->rows, B->cols, A->cols);
#endif

	}*/

	template<typename T>
	struct GemmImp {
		T *packA = nullptr, *packB = nullptr;
		static constexpr int KernelRowSize = 16/sizeof(T)*sizeof(float);
		static constexpr int KernelColSize = 4;

		~GemmImp() {
			if (packA) aligned_free(packA);
			if (packB) aligned_free(packB);
		}

		FORCE_INLINE void GEMM_Kernel(T* a,const int& lda,T* b,const int& ldb,T* c,const int ldc,int m,int n,int k, bool start) {
			//alloc memory for packing
			const std::size_t packedARows = m % KernelRowSize == 0 ? m : (m + KernelRowSize - m % KernelRowSize);
			const std::size_t packedBCols = n % KernelColSize == 0 ? n : (n + KernelColSize - n % KernelColSize);

			if (!packA && !packB) {
				packA = mynew<T>(packedARows * k, VECTORIZATION_ALIGN_BYTES);
				packB = mynew<T>(packedBCols * k, VECTORIZATION_ALIGN_BYTES);
			}
			//pack InnerKernel A(rm*rk) into packedA
			PackLhs(a, lda, m, k, packA, KernelRowSize, KernelRowSize);
			if (start)
				PackRhs(b, ldb, k, n, packB, KernelColSize, KernelColSize);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(packA, KernelRowSize, packedARows / KernelRowSize * k, "packA: ");
			DEBUG_TOOLS::printRawMatrix(packB, KernelColSize, packedBCols / KernelColSize * k, "packB: ");
			DEBUG_TOOLS::printRawMatrix(a, m, k, "A:");
			DEBUG_TOOLS::printRawMatrix(b, k, n, "B:");
#endif // _DEBUG


			GemmInnerLoop(packA, KernelRowSize, packB, KernelColSize, c, ldc, m, n, k, KernelRowSize, KernelColSize);
		}
	};

	template<typename T>
	void Gemm(T* A, const int& lda, T* B, int const& ldb, T* C, int const ldc, int m, int n, int k) {
		int colStep = k_kernel, rowStep = m_kernel;
		auto handle = GemmImp<float>();
		for (int colIndex = 0; colIndex < k; colIndex += colStep) {
			colStep = _min_(k - colIndex, k_kernel);
			for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
				rowStep = _min_(m - rowIndex, m_kernel);
				handle.GEMM_Kernel(A + rowIndex + colIndex * lda, lda, B + colIndex, ldb, C + rowIndex, ldc, rowStep, n, colStep, rowIndex == 0);
			}
		}
	}

}  // namespace CSM
