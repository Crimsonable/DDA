#pragma once
#include "forwardDecleration.h"
#include "Product.h"
#include "CwiseOpImp.h"


namespace Functions
{
	struct FunctionBase
	{
		template<typename ...Args>
		static void apply(Args ...args) {}
	};

	struct MatMulOp: public FunctionBase
	{
		template<typename lhs_tensor,typename rhs_tensor, typename dst_tensor>
		static void apply(lhs_tensor* l, rhs_tensor* r, dst_tensor* d) {
			DDA::Product(l, r, d);
		}
	};

	struct CwiseSumOp :public FunctionBase
	{
		template<typename T>
		static auto func(T&& l, T&& r) { return l + r; }

		template<typename lhs_tensor, typename rhs_tensor, typename dst_tensor>
		static void apply(lhs_tensor* l, rhs_tensor* r, dst_tensor* d) {
			CPU_OP::CwiseBasicImp<lhs_tensor, rhs_tensor, dst_tensor, CwiseSumOp>(l, r, d);
		}
	};

	struct CwiseSubOp :public FunctionBase
	{
		template<typename T>
		static auto func(T&& l, T&& r) { return l - r; }

		template<typename lhs_tensor, typename rhs_tensor, typename dst_tensor>
		static void apply(lhs_tensor* l, rhs_tensor* r, dst_tensor* d) {
			CPU_OP::CwiseBasicImp<lhs_tensor, rhs_tensor, dst_tensor, CwiseSubOp>(l, r, d);
		}
	};

	struct CwiseMulOp :public FunctionBase
	{
		template<typename T>
		static auto func(T&& l, T&& r) { return l * r; }

		template<typename lhs_tensor, typename rhs_tensor, typename dst_tensor>
		static void apply(lhs_tensor* l, rhs_tensor* r, dst_tensor* d) {
			CPU_OP::CwiseBasicImp<lhs_tensor, rhs_tensor, dst_tensor, CwiseMulOp>(l, r, d);
		}
	};
}