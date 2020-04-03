#pragma once
#include "SIMD.h"
#include "forwardDecleration.h"
#include "memory.h"
#include "ProductHandler.h"
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

namespace DDA {
	
	template<typename T>
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

		FORCE_INLINE void GEMM_Kernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst) {
			//alloc memory for packing
			static std::size_t packedARows = rm % KernelRowSize == 0 ? rm : (rm + KernelRowSize - rm % KernelRowSize);
			static std::size_t packedBCols = n % 4 == 0 ? n : (n + 4 - n % 4);

			if (!packedA && !packedB) {
				packedA = mynew<T>(packedARows * rk, VECTORIZATION_ALIGN_BYTES);
				packedB = mynew<T>(packedBCols * rk, VECTORIZATION_ALIGN_BYTES);
			}
			if (!handler) {
				handler = new ProductHanlder<T>(m, n, k, rm, rk, rn, packedARows, packedBCols, packedA, packedB, C);
			}

			handler->update(rm, rn, rk, packedBCols, C);

			//pack InnerKernel A(rm*rk) into packedA
			PackMatrixA_final(b_A, handler->GetInnerRows(), m, rk, rm, handler->GetPadRows(), handler->GetTotalRows(), handler->GetPadStep(), packedA);
			if (isFirst)
				PackMatrixB_final(b_B, handler->GetInnerCols(), n, rk, k, packedBCols - n, packedB);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(packedA, handler->GetInnerRows(), packedARows*rk/ handler->GetInnerRows(), "packedA:");
			DEBUG_TOOLS::printRawMatrix(packedB, rk, packedBCols, "packedB:");
			DEBUG_TOOLS::printRawMatrix(b_A, m, k, "A:");
			DEBUG_TOOLS::printRawMatrix(b_B, k, n, "B:");
#endif // DEBUG_INFO

			handler->GEMM_InnerLoop();
		}

		template<typename T>
		FORCE_INLINE void GEMP_Kernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k) {
			static std::size_t packedARows = rm % KernelRowSize == 0 ? rm : (rm + KernelRowSize - rm % KernelRowSize);

			if (!packedA && !packedB) {
				packedA = mynew<T>(packedARows * rk, VECTORIZATION_ALIGN_BYTES);
			}
			if (!handler) {
				handler = new ProductHanlder<T>(m, n, k, rm, rk, rn, packedARows, rn, packedA, b_B, C);
			}

			handler->update(rm, rn, rk, rn, C);

			//pack InnerKernel A(rm*rk) into packedA
			PackMatrixA_final(b_A, handler->GetInnerRows(), m, rk, rm, handler->GetPadRows(), handler->GetTotalRows(), handler->GetPadStep(), packedA);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(packedA, handler->GetInnerRows(), packedARows*rk / handler->GetInnerRows(), "packedA:");
			DEBUG_TOOLS::printRawMatrix(b_B, rk, n, "packedB:");
			DEBUG_TOOLS::printRawMatrix(b_A, m, k, "A:");
			DEBUG_TOOLS::printRawMatrix(b_B, k, n, "B:");
#endif // DEBUG_INFO

			handler->GEMP_InnerLoop();
		}
	};

	template <typename lhs, typename rhs, typename dst>
	void Product(lhs *A, rhs *B, dst *C) {
		auto ptrA = A->data();
		auto ptrB = B->data();
		auto ptrC = C->data();
		int m = A->rows, k = A->cols, n = B->cols;

#ifndef HAS_CUDA
		int colStep = k_kernel, rowStep = m_kernel;
		ProductCls<std::remove_reference_t<decltype(*ptrA)>> productInstance;

			if (n > 64) {
				for (int colIndex = 0; colIndex < k; colIndex += colStep) {
					colStep = _min_(k - colIndex, k_kernel);
					for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
						rowStep = _min_(m - rowIndex, m_kernel);
						productInstance.GEMM_Kernel(ptrA + m * colIndex + rowIndex, ptrB + colIndex, ptrC + rowIndex, rowStep, colStep, n, m, n, k, rowIndex == 0);
					}
				}
			}
			else {
				for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
					rowStep = _min_(m - rowIndex, m_kernel);
					productInstance.GEMP_Kernel(ptrA + rowIndex, ptrB, ptrC + rowIndex, rowStep, k, n, m, n, k);
				}
			}
#else
		cudaProduct(ptrA, ptrB, ptrC, A->rows, B->cols, A->cols);
#endif
		
	}

	/*template <typename lhs, typename rhs, typename dst>
	void Product(lhs *A, rhs *B, dst *C, int kkernel) {
		auto ptrA = A->data();
		auto ptrB = B->data();
		auto ptrC = C->data();
		int m = A->rows, k = A->cols, n = B->cols;

#ifndef HAS_CUDA
		int p, q, rp = kkernel, rq = m_kernel;
		ProductCls<std::remove_reference_t<decltype(*ptrA)>> productInstance;

		if (n != 1) {
			for (p = 0; p < k; p += rp) {
				rp = _min_(k - p, kkernel);
				for (q = 0; q < m; q += rq) {
					rq = _min_(m - q, m_kernel);
					productInstance.InnerKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0);
				}
			}
		}
		else {
			for (p = 0; p < k; p += rp) {
				rp = _min_(k - p, kkernel);
				for (q = 0; q < m; q += rq) {
					rq = _min_(m - q, m_kernel);
					productInstance.MatDotVectorKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - rp && q == m - rq);
				}
			}
		}
#else
		cudaProduct(ptrA, ptrB, ptrC, A->rows, B->cols, A->cols);
#endif
	}*/
}  // namespace DDA