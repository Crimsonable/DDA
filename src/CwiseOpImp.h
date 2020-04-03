#pragma once
#include "Matrix.h"
#include "SIMD.h"
using namespace DDA::SSE_OP;

namespace CPU_OP {

	template<typename lhs,typename rhs,typename dst, typename Caller>
	void CwiseBasicImp(lhs* l, rhs* r, dst* d) {
		auto lhs_dataptr = l->data();
		auto rhs_dataptr = r->data();
		auto dst_dataptr = d->data();
		using scalar = typename std::remove_reference_t<decltype(*lhs_dataptr)>;
		static constexpr int main_step = 8 / sizeof(float) * sizeof(scalar);
		int size = l->size, endVec = size - size % main_step;
		v_256<scalar> l_vec, r_vec;

		for (int index = 0; index < endVec; index += main_step) {
			load_ps(l_vec.v, lhs_dataptr + index);
			load_ps(r_vec.v, rhs_dataptr + index);
			store(dst_dataptr + index, Caller::func(l_vec.v, r_vec.v));
		}

		for (int index = endVec; index < size; ++index) {
			*(dst_dataptr + index) = Caller::func(*(lhs_dataptr + index), *(rhs_dataptr + index));
		}
	}
}