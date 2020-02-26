#pragma once
#include "forwardDecleration.h"
#include "ProductImp.h"

namespace DDA {
	//using namespace DDA::SSE_OP;
	template<typename T>
	class ProductHanlder {
	private:
		static constexpr int InnerKernelRows = 16 / (sizeof(T) / sizeof(float));
		static constexpr int InnerKernelCols = 4;
		//static constexpr int InnerKernelCols = 8;
		static constexpr int mainStep = InnerKernelRows;
		int padStep = 0, leftStep, EndVec, padRows = 0, rm, rk, rn, packedARows, packedBCols, m, n, k;
		T *packedA_ptr, *packedB_ptr, *C_ptr;
	public:
		ProductHanlder() {}
		ProductHanlder(int m, int n, int k, int rm, int rk, int rn, int packedARows, int packedBCols, T *packedA_ptr, T *packedB_ptr, T *C_ptr) {
			this->m = m;
			this->n = n;
			this->k = k;
			this->rm = rm;
			this->rk = rk;
			this->rn = rn;
			this->packedARows = packedARows;
			this->packedBCols = packedBCols;
			this->packedA_ptr = packedA_ptr;
			this->packedB_ptr = packedB_ptr;
			this->C_ptr = C_ptr;
		}

		FORCE_INLINE void update(int rm, int rn, int rk, int packedBCols, T *C_ptr) {
			this->rm = rm;
			this->rn = rn;
			this->rk = rk;
			this->packedBCols = packedBCols;
			this->C_ptr = C_ptr;
			leftStep = rm % InnerKernelRows;
			EndVec = rm - leftStep;
			if constexpr (std::is_same_v<T, float>) {
				if (leftStep > 4) {
					padStep = 8;
					leftStep = leftStep % 8;
				}
				else if (leftStep) {
					padStep = 4;
					leftStep = leftStep % 4;
				}
			}
			else if constexpr (std::is_same_v<T, double>) {
				if (leftStep) {
					if (leftStep <= 4 && leftStep) {
						padStep = 4;
						leftStep = leftStep % 4;
					}
					else {
						padStep = 8;
						leftStep = leftStep % 8;
					}
				}
			}
			if constexpr (std::is_same_v<T, double>) {
				if(padStep==8)
					EndVec = EndVec > 0 ? EndVec : 8;
				else
					EndVec = EndVec > 0 ? EndVec : 0;
			}
			else if constexpr(std::is_same_v<T,float>)
				EndVec = EndVec > 0 ? EndVec : 0;
			if (padStep)
				this->padRows = leftStep ? padStep - leftStep : 0;
			else
				this->padRows = 0;
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

		template<typename D,std::enable_if_t<std::is_same_v<D, float>,int> = 0>
		inline void InnerLoop() {
#pragma omp parallel
			{
#pragma omp for schedule(dynamic) nowait
				for (int j = 0; j < packedBCols; j += InnerKernelCols) {
					for (int i = 0; i < EndVec; i += InnerKernelRows) {
						AddDot8x4<T, v_256<T>>(packedA_ptr + i * rk, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + InnerKernelRows > rm);
					}
					for (int i = EndVec; i < rm; i += padStep) {
						if (padStep > 4)
							AddDot4x4<T, v_256<T>>(packedA_ptr + rk * i, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + padStep > rm);
						else
							AddDot4x4<T, v_128<T>>(packedA_ptr + rk * i, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + padStep > rm);
					}
				}
			}
		}

		/*template<typename D, std::enable_if_t<std::is_same_v<D, float>, int> = 0>
		inline void InnerLoop() {
#pragma omp parallel //shared(packedA_ptr,packedB_ptr,C_ptr,rk,m,n,k,padStep,EndVec,rm)
			{
#pragma omp for schedule(dynamic) nowait
				for (int j = 0; j < packedBCols; j += InnerKernelCols) {
					for (int i = 0; i < rm; i += InnerKernelRows) {
						AddDot8x8_ps(packedA_ptr + rk * i, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + InnerKernelRows > rm);
					}
				}
			}
		}*/

		template<typename D, std::enable_if_t<std::is_same_v<D, double>,int> = 0>
		inline void InnerLoop() {
#pragma omp parallel //shared(packedA_ptr,packedB_ptr,C_ptr,rk,m,n,k,padStep,EndVec,rm)
			{
#pragma omp for schedule(dynamic) nowait
				for (int j = 0; j < packedBCols; j += InnerKernelCols) {
					for (int i = 0; i < EndVec; i += InnerKernelRows) {
						AddDot8x4<T, v_256<T>>(packedA_ptr + rk * i, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + InnerKernelRows > rm);
					}
					for (int i = EndVec; i < rm; i += padStep) {
						AddDot4x4<T, v_256<T>>(packedA_ptr + rk * i, packedB_ptr + j * rk, C_ptr + i + j * m, rk, m, n, k, j, padRows, i + InnerKernelRows > rm);
					}
				}
			}
		}
	};
}