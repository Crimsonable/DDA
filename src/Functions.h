#pragma once
#include "forwardDecleration.h"
#include "Gemm.h"
#include "CwiseOpImp.h"

using namespace CSM::internal;

namespace Functions
{
	struct DefaultImp {
		struct FunctionBase
		{
			template<typename ...Args>
			static void apply(Args ...args) {}
		};

		struct MatMulOp : public FunctionBase
		{
			template<typename T>
			FORCE_INLINE static void apply(T* l, const int& lda, T* r, const int& ldb, T* d, const int& ldc, int m, int n, int k) {
				CSM::Gemm(l, lda, r, ldb, d, ldc, m, n, k);
			}
		};

		struct MatAddOpAccumulate :public FunctionBase
		{
			template<typename T>
			static auto func(T&& l, T&& r) { return l + r; }

			template<typename T>
			static void apply(T* l, const int& lda, T* d, const int& ldc, int rows, int cols) {
				CPU_OP::CwiseBasicImp<T, MatAddOpAccumulate>(l, lda, d, ldc, d, ldc, rows, cols);
			}
		};

		struct MatAddOp :public FunctionBase
		{
			template<typename T>
			static auto func(T&& l, T&& r) { return l + r; }

			template<typename T>
			static void apply(T* l, const int& lda, T* r, const int& ldb, T* d, const int& ldc, int rows, int cols) {
				CPU_OP::CwiseBasicImp<T, MatAddOp>(l, lda, r, ldb, d, ldc, rows, cols);
			}
		};

		struct MatSubOp :public FunctionBase
		{
			template<typename T>
			static auto func(T&& l, T&& r) { return l - r; }

			template<typename T>
			static void apply(T* l, const int& lda, T* r, const int& ldb, T* d, const int& ldc, int rows, int cols) {
				CPU_OP::CwiseBasicImp<T, MatSubOp>(l, lda, r, ldb, d, ldc, rows, cols);
			}
		};

		struct CwiseMulOp :public FunctionBase
		{
			template<typename T>
			static auto func(T&& l, T&& r) { return l * r; }

			template<typename T>
			static void apply(T* l, const int& lda, T* r, const int& ldb, T* d, const int& ldc, int rows, int cols) {
				CPU_OP::CwiseBasicImp<T, CwiseMulOp>(l, lda, r, ldb, d, ldc, rows, cols);
			}
		};

		struct TransposeOp :public FunctionBase
		{
			template<typename T>
			static void apply(T* s, int ld1, T* d, int ld2, int rows, int cols) {
				CPU_OP::transpose(s, ld1, d, ld2, rows, cols);
			}
		};
	};

	/*struct CwiseScalarMulOp:public FunctionBase
	{
		template<typename T>
		static auto func(T&& mul, T&& add, T&& r) {
			return mul * r + add;
		}

		template<typename rhs_tensor, typename dst_tensor, typename T = typename traits<dst_tensor>::scalar >
		static void apply(T mul, T add, rhs_tensor* r, dst_tensor* d, int offset_r, int offset_dst, int rows, int cols) {
			CPU_OP::CwiseImpSingleOp<T, CwiseMulOp>(mul, add, r->data() + offset_r, d->data() + offset_dst, r->rows, d->rows, rows, cols);
		}
	};*/
}