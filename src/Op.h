#pragma once
#include "forwardDecleration.h"
#include "Functions.h"

#define OP_TRAITS_PROPERTIES static constexpr bool lXpr = internal::traits<lhs_type>::isXpr;\
							 static constexpr bool rXpr = internal::traits<rhs_type>::isXpr;\
							 static constexpr bool islXprDot = internal::traits<lhs_type>::isDot;\
							 static constexpr bool isrXprDot = internal::traits<rhs_type>::isDot

#define IS_BASE_OF_COMMONBASE(_T1,_T2) std::enable_if<std::is_base_of_v<CommonBase,##_T1##> && std::is_base_of_v<CommonBase,##_T2##>,int>::type=0

#define OP_PREDEFINE_TRAITS(_T1,_T2) static constexpr int size = traits<##_T1##>::size;\
											using scalar = typename traits<##_T1##>::scalar;\
											using lhs_type = ##_T1##;\
											using rhs_type = ##_T2##;\
											static constexpr bool isXpr = true

namespace DDA
{
	namespace internal
	{
		/*template<>
		struct traits<float>
		{
			static constexpr bool isXpr = false;
			static constexpr bool isDot = false;
		};

		template<>
		struct traits<double>
		{
			static constexpr bool isXpr = false;
			static constexpr bool isDot = false;
		};

		template <typename Scalar, typename lhs, typename rhs>
		struct traits<CwiseOpsum<Scalar, lhs, rhs>>
		{
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			using lhs_type = lhs;
			using rhs_type = rhs;
			static constexpr bool isXpr = true;
			static constexpr bool isDot = false;
		};

		template <typename Scalar, typename lhs, typename rhs>
		struct traits<CwiseOpproduct<Scalar, lhs, rhs>>
		{
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			using lhs_type = lhs;
			using rhs_type = rhs;
			static constexpr bool isXpr = true;
			static constexpr bool isDot = false;
		};*/

		template <typename Lhs, typename Rhs, typename Function>
		struct traits<CwiseBaseOp<Lhs, Rhs, Function>>{
			OP_PREDEFINE_TRAITS(Lhs, Rhs);
			static constexpr bool isDot = false;
		};

		template <typename Lhs, typename Rhs>
		struct traits<CwiseSumOp<Lhs, Rhs>> {
			OP_PREDEFINE_TRAITS(Lhs, Rhs);
			static constexpr bool isDot = false;
		};

		template <typename Lhs, typename Rhs>
		struct traits<CwiseSubOp<Lhs, Rhs>> {
			OP_PREDEFINE_TRAITS(Lhs, Rhs);
			static constexpr bool isDot = false;
		};

		template <typename Lhs, typename Rhs>
		struct traits<MatMulOp<Lhs, Rhs>>{
			OP_PREDEFINE_TRAITS(Lhs, Rhs);
			static constexpr bool isDot = true;
		};
	} // namespace internals

	template<typename Derived>
	class OpBase:CommonBase
	{
	protected:
		using lhs_type = typename internal::traits<Derived>::lhs_type;
		using rhs_type = typename internal::traits<Derived>::rhs_type;
		using scalar = typename internal::traits<Derived>::scalar;
		using TemporaryType = Matrix<scalar, -1, -1>;
		OP_TRAITS_PROPERTIES;
	public:
		lhs_type* lhs;
		rhs_type* rhs;

		OpBase() { }
		~OpBase() {	}

		Derived* derived()
		{
			static_cast<Derived*>(this);
		}

		template<typename Dst,typename Src>
		void run(Dst *d, const Src &s, bool force_lazy)
		{
			Src *temp_src = const_cast<Src*>(&s);
			temp_src->dispatch(d, force_lazy);
		}
	};

	template<typename Lhs,typename Rhs>
	class MatMulOp :public OpBase<MatMulOp<Lhs, Rhs>>
	{
	protected:
		typedef MatMulOp<Lhs, Rhs> Self;
		typedef OpBase<Self> Base;
		using scalar = typename internal::traits<Self>::scalar;
		using TemporaryType = Matrix<scalar, -1, -1>;
		TemporaryType *tempRes = nullptr, *tempLhs = nullptr, *tempRhs = nullptr;
		bool need_lazy_eval, need_lhs_eval, need_rhs_eval;
	public:
		int rows, cols;

		MatMulOp(const Lhs& l, const Rhs& r)
		{
			this->lhs = const_cast<Lhs*>(&l);
			this->rhs = const_cast<Rhs*>(&r);
			need_lazy_eval = false;
			rows = l.rows;
			cols = r.cols;
		}

		~MatMulOp() {
			if (!need_lazy_eval) delete tempRes;
		}

		template<typename Dst>
		inline constexpr void EvaluationPolicy(Dst* d, bool force_lazy)
		{
			if constexpr (this->lXpr)
			{
				need_lhs_eval = true;
				tempLhs = new TemporaryType;
				tempLhs->alias() = *(this->lhs);
			}
			else tempLhs = this->lhs;
			if constexpr (this->rXpr)
			{
				need_rhs_eval = true;
				tempRhs = new TemporaryType;
				tempRhs->alias() = *(this->rhs);
			}
			else tempRhs = this->rhs;
			if (need_lazy_eval || force_lazy) {
				tempRes = d;
				need_lazy_eval = true;
			}
			else {
				tempRes = new TemporaryType;
				tempRes->resize(rows, cols);
			}
		}
	
		template<typename Dst>
		void dispatch(Dst* d, bool force_lazy)
		{
			EvaluationPolicy(d, force_lazy);
			Functions::MatMulOp::apply(tempLhs, tempRhs, tempRes);
			if constexpr (this->lXpr) delete tempLhs;
			if constexpr (this->rXpr) delete tempRhs;
			if (!force_lazy) d->swap(tempRes);
		}

		inline Base &toXprBase()
		{
			return *static_cast<Base*>(this);
		}
	};

	template<typename Lhs,typename Rhs, typename Function>
	class CwiseBaseOp :public OpBase<CwiseBaseOp<Lhs, Rhs, Function>>
	{
	protected:
		typedef CwiseBaseOp<Lhs, Rhs, Function> Self;
		typedef OpBase<Self> Base;
		using scalar = typename internal::traits<Self>::scalar;
		using TemporaryType = Matrix<scalar, -1, -1>;
		TemporaryType *tempRes = nullptr, *tempLhs = nullptr, *tempRhs = nullptr;
		bool need_lazy_eval, need_lhs_eval = false, need_rhs_eval = false;
	public:
		int rows, cols;

		CwiseBaseOp(const Lhs& l, const Rhs& r) {
			this->lhs = const_cast<Lhs*>(&l);
			this->rhs = const_cast<Rhs*>(&r);
			need_lazy_eval = true;
			rows = l.rows;
			cols = r.cols;
		}

		~CwiseBaseOp() {
			if (need_lhs_eval) delete tempLhs;
			if (need_rhs_eval) delete tempRhs;
			if (!need_lazy_eval) delete tempRes;
		}

		template<typename Dst>
		inline constexpr void EvaluationPolicy(Dst* d, bool needLazy)
		{
			if constexpr (this->lXpr) {
				if constexpr (this->islXprDot) {
					need_lhs_eval = true;
					tempLhs = new TemporaryType;
					tempLhs->alias() = *this->lhs;
				}
				else {
					static_cast<Lhs*>(this->lhs)->run(d, *this->lhs, true);
					tempLhs = d;
				}
			}
			else
				tempLhs = this->lhs;
			if constexpr (this->rXpr) {
				need_rhs_eval = true;
				tempRhs = new TemporaryType;
				tempRhs->alias() = *this->rhs;
			}
			else
				tempRhs = this->rhs;
			if(need_lazy_eval)
				tempRes = d;
			else {
				tempRes = new TemporaryType;
				tempRes->resize(rows, cols);
			}
		}

		template<typename Dst>
		void dispatch(Dst* d, bool needLazy)
		{
			EvaluationPolicy(d, needLazy);
			Function::apply(tempLhs, tempRhs, tempRes);
		}

		inline Base &toXprBase()
		{
			return *static_cast<Base*>(this);
		}
	};

	template<typename Lhs, typename Rhs>
	class CwiseSumOp :public CwiseBaseOp<Lhs, Rhs, typename Functions::CwiseSumOp>
	{
	public:
		CwiseSumOp(const Lhs& l, const Rhs& r) :CwiseBaseOp<Lhs, Rhs, typename Functions::CwiseSumOp>(l, r) {}
	};

	template<typename Lhs,typename Rhs>
	class CwiseSubOp :public CwiseBaseOp<Lhs, Rhs, typename Functions::CwiseSubOp>
	{
	public:
		CwiseSubOp(const Lhs& l, const Rhs& r) :CwiseBaseOp<Lhs,Rhs,typename Functions::CwiseSubOp>(l,r){}
	};

	template<typename Lhs, typename Rhs>
	class CwiseMulOp :public CwiseBaseOp<Lhs, Rhs, typename Functions::CwiseMulOp>
	{
		CwiseMulOp(const Lhs& l, const Rhs& r) :CwiseBaseOp<Lhs, Rhs,typename Functions::CwiseMulOp>(l, r) {}
	};

	template <typename Derived,
			  typename otherDerived, 
		      typename RetType = typename MatMulOp<Derived, otherDerived>,
			  typename IS_BASE_OF_COMMONBASE(Derived, otherDerived)>
	RetType operator*(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::MatMulOp(l, r));
	}

	template <typename Derived, 
			  typename otherDerived,
		      typename RetType = typename CwiseSumOp<Derived, otherDerived>,
			  typename IS_BASE_OF_COMMONBASE(Derived, otherDerived)>
	RetType operator+(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::CwiseSumOp(l, r));
	}

	template <typename Derived,
		typename otherDerived,
		typename RetType = typename CwiseSubOp<Derived, otherDerived>,
		typename IS_BASE_OF_COMMONBASE(Derived, otherDerived)>
		RetType operator-(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::CwiseSubOp(l, r));
	}

	/*template <typename Derived,
		typename otherDerived,
		typename RetType = typename CwiseMulOp<Derived, otherDerived>,
		typename IS_BASE_OF_COMMONBASE(Derived, otherDerived)>
		RetType operator+(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::CwiseMulOp(l, r));
	}*/
}
