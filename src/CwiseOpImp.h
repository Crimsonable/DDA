#pragma once
#include "Matrix.h"
#include "SIMD.h"
using namespace CSM::SSE_OP;

namespace CPU_OP {

	template<typename T, typename Caller>
	void CwiseBasicImp(T* lhs_dataptr, const int& lda, T* rhs_dataptr, const int& ldb, T* dst_dataptr, const int& ldc, const int& rows, const int& cols) {
		static constexpr int main_step = 8 / sizeof(float) * sizeof(T);
		int endVec = rows - rows % main_step;
		v_256<T> l_vec, r_vec;

#pragma omp parallel private(l_vec,r_vec)
		{
#pragma omp for schedule(static) nowait
			for (int col_index = 0; col_index < cols; ++col_index) {
				int offset_l = lda * col_index, offset_r = ldb * col_index, offset_dst = ldc * col_index;
				for (int row_index = 0; row_index < endVec; row_index += main_step) {
					load_ps(l_vec.v, lhs_dataptr + offset_l + row_index);
					load_ps(r_vec.v, rhs_dataptr + offset_r + row_index);
					store(dst_dataptr + offset_dst + row_index, Caller::func(l_vec.v, r_vec.v));
				}
				for (int row_index = endVec; row_index < rows; ++row_index) {
					*(dst_dataptr + offset_dst + row_index) = Caller::func(*(lhs_dataptr + offset_l + row_index), *(rhs_dataptr + offset_r + row_index));
				}
			}
		}
	}

	template<typename T, typename Caller>
	void CwiseImpSingleOp(T mul, T add, T* rhs_dataptr, T* dst_dataptr, const int& ldb, const int& ldc, const int& rows, const int& cols) {
		static constexpr int main_step = 8 / sizeof(float) * sizeof(T);
		int endVec = rows - rows % main_step;
		v_256<T> mul_vec, add_vec, r_vec;
		load_ps1(mul_vec.v, &mul);
		load_ps1(add_vec.v, &add);

#pragma omp parallel private(r_vec)
		{
#pragma omp for schedule(static) nowait
			for (int col_index = 0; col_index < cols; ++col_index) {
				int offset_r = ldb * col_index, offset_dst = ldc * col_index;
				for (int row_index = 0; row_index < endVec; row_index += main_step) {
					load_ps(r_vec.v, rhs_dataptr + offset_r + row_index);
					store(dst_dataptr + offset_dst + row_index, Caller::func(mul_vec.v, add_vec.v, r_vec.v));
				}
				for (int row_index = endVec; row_index < rows; ++row_index) {
					*(dst_dataptr + offset_dst + row_index) = Caller::func(mul, add, *(rhs_dataptr + offset_r + row_index));
				}
			}
		}
	}
}