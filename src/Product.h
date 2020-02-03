#pragma once
#include "SIMD.h"
#include "forwardDecleration.h"
#include "memory.h"
#include "ProductHandler.h"

#ifdef _DEBUG
#include"testTools.h"
#endif // _DEBUG


#define m_kernel 128
#define k_kernel 128
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

		void PackMatrixA(T *A, int blockrows, int m, int block_k, T *packA, int pad_rows) {
			int i = 0, j = 0;
			for (j = 0; j < block_k; ++j) {
				T *ptr = A + j * m;
				memcpy(packA + j * blockrows, ptr, sizeof(T)*(blockrows - pad_rows));
				for (int k = 0; k < pad_rows; ++k) {
					*(packA + i + k + j * blockrows) = 0;
				}
			}
		}

		inline void PackMatrixB(T *B, int blockcols, int block_k, int k, T *packB) {
			constexpr int step = 8 / (sizeof(T) / sizeof(float));
			int EndVec = block_k - block_k % step;
			for (int j = 0; j < blockcols; ++j) {
				memcpy(packB + j * block_k, B + j * k, sizeof(T)*block_k);
			}
		}

		inline void PackMatrixB_pad(int blockcols, int block_k, T *packB) {
			for (int j = 0; j < blockcols; ++j) {
				for (int i = 0; i < block_k; ++i) {
					*(packB + i + j * block_k) = 0;
				}
			}
		}

		inline void PackMatrixA_final(T *A, T *packedA, int InnerKernel_rows, int m, int rk, int rm, int pad_rows, int after_pad, int padStep) {
			int EndVec = after_pad - after_pad % InnerKernel_rows;
#pragma omp parallel shared(A,packedA)
			{
#pragma omp for schedule(dynamic) nowait
				for (int i = 0; i < EndVec; i += InnerKernel_rows) {
					if (pad_rows && rm - i < InnerKernel_rows)
						PackMatrixA(A + i, InnerKernel_rows, m, rk, packedA + rk * i, pad_rows);
					else
						PackMatrixA(A + i, InnerKernel_rows, m, rk, packedA + rk * i, 0);
				}
				for (int i = EndVec; i < rm; i += padStep) {
					if (pad_rows && rm - i < padStep)
						PackMatrixA(A + i, padStep, m, rk, packedA + rk * i, padStep - rm + i);
					else
						PackMatrixA(A + i, padStep, m, rk, packedA + rk * i, 0);
				}
			}
		}

		inline void PackMatrixB_final(T *B, T *packedB, int InnerKernel_cols, int n, int rk, int k, int pad_cols) {
#pragma omp parallel shared(B,packedB)
			{
#pragma omp for schedule(dynamic) nowait
				for (int j = 0; j < n; j += InnerKernel_cols)
					PackMatrixB(B + j * k, InnerKernel_cols, rk, k, packedB + j * rk);
				if (pad_cols)
					PackMatrixB_pad(pad_cols, rk, packedB + n * rk);
			}
		}

	public:
		~ProductCls() {
			aligned_free(packedA);
			aligned_free(packedB);
			std::free(handler);
		}

		void InnerKernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst, bool isLast) {
			//alloc memory for packing
			std::size_t packedARows = rm % 8 == 0 ? rm : (rm + 8 - rm % 8);
			std::size_t packedBCols = n % 4 == 0 ? n : (n + 4 - n % 4);
			if (!packedA && !packedB) {
				packedA = mynew<T>(packedARows * rk, 16);
				packedB = mynew<T>(packedBCols * rk, 16);
			}
			if (!handler) {
				handler = new ProductHanlder<T>(m, n, k, rm, rk, rn, packedARows, packedBCols, packedA, packedB, C);
			}

			handler->update(rm, rn, rk, packedBCols, C);

			//pack InnerKernel A(rm*rk) into packedA
			PackMatrixA_final(b_A, packedA, handler->GetInnerRows(), m, rk, rm, handler->GetPadRows(), handler->GetTotalRows(), handler->GetPadStep());
			if (isFirst)
				PackMatrixB_final(b_B, packedB, handler->GetInnerCols(), n, rk, k, packedBCols - n);

#ifdef DEBUG_INFO
			DEBUG_TOOLS::printRawMatrix(packedA, handler->GetTotalRows(), rk, "packedA:");
			DEBUG_TOOLS::printRawMatrix(packedB, rk, packedBCols, "packedB:");
			DEBUG_TOOLS::printRawMatrix(b_A, m, k, "A:");
			DEBUG_TOOLS::printRawMatrix(b_B, k, n, "B:");
#endif // DEBUG_INFO

			handler->template InnerLoop<T>();
		}

		void MatDotVectorKernel(T *b_A, T *b_B, T *C, int rm, int rk, int rn, int m, int n, int k, bool isFirst, bool isLast) {
			const constexpr int InnerKernel_rows = 4 * inner_rows / sizeof(T);

			std::size_t packedARows = rm % 4 == 0 ? rm : (rm + 4 - rm % 4);
			static T *packedA = mynew<T>(packedARows * rk, 16);
			int EndVecRows = rm - rm % InnerKernel_rows;

			PackMatrixA_final(b_A, packedA, InnerKernel_rows, m, rk, rm, packedARows - rm, packedARows, 0);

#pragma omp parallel for
			for (int i = 0; i < EndVecRows; i += InnerKernel_rows) {
				AddDot8x1<T, v_256<T>>(packedA + rk * i, b_B, C + i, rk, m, 1, k);
			}
			for (int i = EndVecRows; i < rm; i += basic_step) {
				AddDot4x1<T, v_128<T>>(packedA + rk * i, b_B, C + i, rk, m, 1, k);
			}

			if (isLast) {
				aligned_free(packedA);
			}
		}
	};

    template <typename lhs, typename rhs, typename dst>
    void Product(lhs *A, rhs *B, dst *C) {
        auto ptrA = A->data();
        auto ptrB = B->data();
        auto ptrC = C->data();
        int p, q, rp = k_kernel, rq = m_kernel;
        int m = A->rows, k = A->cols, n = B->cols;
		ProductCls<std::remove_reference_t<decltype(*ptrA)>> productInstance;

        if (n != 1) {
            for (p = 0; p < k; p += rp) {
                rp = _min_(k - p, k_kernel);
                for (q = 0; q < m; q += rq) {
                    rq = _min_(m - q, m_kernel);
                    productInstance.InnerKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - rp && q == m - rq);
                }
            }
        } else {
            for (p = 0; p < k; p += rp) {
                rp = _min_(k - p, k_kernel);
                for (q = 0; q < m; q += rq) {
                    rq = _min_(m - q, m_kernel);
                    productInstance.MatDotVectorKernel(ptrA + m * p + q, ptrB + p, ptrC + q, rq, rp, n, m, n, k, q == 0, p == k - rp && q == m - rq);
                }
            }
        }
    }
}  // namespace DDA