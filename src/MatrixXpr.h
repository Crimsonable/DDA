#pragma once
#include "MatrixBase.h"
#include "forwardDecleration.h"

namespace DDA {
	namespace internal {
		template<class Scalar,class lhs,class rhs>
		struct traits<CwiseOpsum<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
		};

		template<class Scalar,class lhs,class rhs>
		struct traits<CwiseOpproduct<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
		};

        template<class Scalar,class other>
		struct traits<CwiseOpscalar<Scalar, other>> {
			static constexpr int size = traits<other>::size;
			using scalar = Scalar;
		};
	}

	template<typename lhs,typename rhs>
	class BinaryOp {
	public:
		const lhs* _lhs;
		const rhs* _rhs;
		BinaryOp(const lhs& l, const rhs& r) {
			_lhs = &l;
			_rhs = &r;
		}
		BinaryOp(const lhs* l, const rhs* r) {
			_lhs = l;
			_rhs = r;
		}
	};

	template<typename Scalar, typename lhs,typename rhs>
	class CwiseOpsum: public MatrixBase<CwiseOpsum<Scalar,lhs,rhs>>, public BinaryOp<lhs,rhs>{
	public:
		typedef BinaryOp<lhs, rhs> base;
		CwiseOpsum(){}
		CwiseOpsum(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpsum(const CwiseOpsum& other):base(other._lhs,other._rhs) {}
		//CwiseOpsum(CwiseOpsum&& other):base(other._lhs,other._rh){}

		inline Scalar coffeRef(std::size_t idx) {
			return (*const_cast<lhs*>(this->_lhs)).coffeRef(idx) + (*const_cast<rhs*>(this->_rhs)).coffeRef(idx);
		}
	};

	template<typename Scalar,typename lhs,typename rhs>
	class CwiseOpproduct :public MatrixBase<CwiseOpproduct<Scalar, lhs, rhs>>, public BinaryOp<lhs, rhs> {
	public:
		typedef BinaryOp<lhs, rhs> base;
		CwiseOpproduct(){}
		CwiseOpproduct(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpproduct(const CwiseOpproduct& other):base(other._lhs,other._rhs){}

		inline Scalar coffeRef(std::size_t idx) {
			return (*const_cast<lhs*>(this->_lhs)).coffeRef(idx) * (*const_cast<rhs*>(this->_rhs)).coffeRef(idx);
		}
	};

    template<typename Scalar,typename other>
	class CwiseOpscalar :public MatrixBase<CwiseOpscalar<Scalar, other>>, public BinaryOp<Scalar, other> {
	private:
		Scalar s;
	public:
		typedef BinaryOp<Scalar, other> base;
		CwiseOpscalar(){}
		CwiseOpscalar(const Scalar& s,const other& o):base(0,o),s(s){}
		CwiseOpscalar(const CwiseOpscalar& another):base(nullptr,another._rhs),s(another.s){}

		inline Scalar coffeRef(std::size_t idx) {
			return (*const_cast<other*>(this->_rhs)).coffeRef(idx)*s;
		}
	};

	template<typename Derived, typename otherDerived>
	const CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator +(const Derived& l, const otherDerived& r) {
		return typename CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpsum(l,r);
	}

	template<typename Derived, typename otherDerived, typename std::enable_if<
														(!std::is_arithmetic_v<Derived>)&&(!std::is_arithmetic_v<otherDerived>),int>::type=0>
	const CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator *(const Derived& l, const otherDerived& r) {
		return typename CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpproduct(l, r);
	}

	template<typename Scalar,typename otherDerived,typename std::enable_if<
														std::is_arithmetic_v<Scalar>,int>::type=0>
	const CwiseOpscalar<Scalar,otherDerived> operator *(const Scalar& s,const otherDerived& o){
		return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
	}

	template<typename otherDerived, typename Scalar,typename std::enable_if<
														std::is_arithmetic_v<Scalar>,int>::type=0>
	const CwiseOpscalar<Scalar, otherDerived> operator *(const otherDerived& o, const Scalar& s) {
		return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
	}
}