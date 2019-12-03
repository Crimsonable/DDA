#pragma once
#include "MatrixBase.h"

namespace DDA {
	namespace internal {
		template<typename Scalar,typename lhs,typename rhs>
		struct traits<CwiseOpsum<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
		};

		template<typename Scalar,typename lhs,typename rhs>
		struct traits<CwiseOpproduct<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
		};

		template<typename Scalar,typename other>
		struct traits<CwiseOpscalar<Scalar, other>> {
			static constexpr int size = traits<other>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
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
		static constexpr bool lXpr = internal::traits<lhs>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs>::isXpr;
		CwiseOpsum(){}
		CwiseOpsum(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpsum(const CwiseOpsum& other):base(other._lhs,other._rhs) {}
#ifdef DDA_SIMD
		inline __m128 coffe(std::size_t idx) const {
			if constexpr(lXpr || rXpr) {
				if constexpr (lXpr && rXpr) 
					return _mm_add_ps(this->_lhs->coffe(idx), this->_rhs->coffe(idx));
				else if constexpr(lXpr && !rXpr) 
					return _mm_add_ps(this->_lhs->coffe(idx), _mm_load_ps(&this->_rhs->coffe(idx)));
				else 
					return _mm_add_ps(_mm_load_ps(&this->_lhs->coffe(idx)), this->_rhs->coffe(idx));
			}
			else
				return _mm_add_ps(_mm_load_ps(&this->_lhs->coffe(idx)), _mm_load_ps(&this->_rhs->coffe(idx)));
		}

#else

		inline const Scalar& coffe(std::size_t idx) const {
			return this->_lhs->coffe(idx) + this->_rhs->coffe(idx);
		}
#endif
	};

	template<typename Scalar,typename lhs,typename rhs>
	class CwiseOpproduct :public MatrixBase<CwiseOpproduct<Scalar, lhs, rhs>>, public BinaryOp<lhs, rhs> {
	public:
		typedef BinaryOp<lhs, rhs> base;
		static constexpr bool lXpr = internal::traits<lhs>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs>::isXpr;
		CwiseOpproduct(){}
		CwiseOpproduct(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpproduct(const CwiseOpproduct& other):base(other._lhs,other._rhs){}
#ifdef DDA_SIMD
		inline __m128 coffe(std::size_t idx) const {
			if constexpr (lXpr || rXpr) {
				if constexpr (lXpr && rXpr)
					return _mm_mul_ps(this->_lhs->coffe(idx), this->_rhs->coffe(idx));
				else if constexpr (lXpr && !rXpr)
					return _mm_mul_ps(this->_lhs->coffe(idx), _mm_load_ps(&this->_rhs->coffe(idx)));
				else
					return _mm_mul_ps(_mm_load_ps(&this->_lhs->coffe(idx)), this->_rhs->coffe(idx));
			}
			else
				return _mm_mul_ps(_mm_load_ps(&this->_lhs->coffe(idx)), _mm_load_ps(&this->_rhs->coffe(idx)));
		}
#else
		inline const Scalar& coffe(std::size_t idx) const {
			return this->_lhs->coffe(idx) * this->_rhs->coffe(idx);
		}
#endif
	};

	template<typename Scalar,typename other>
	class CwiseOpscalar :public MatrixBase<CwiseOpscalar<Scalar, other>>, public BinaryOp<Scalar, other> {
	private:
		Scalar s;
	public:
		typedef BinaryOp<Scalar, other> base;
		static constexpr bool rXpr = internal::traits<other>::isXpr;
		CwiseOpscalar(){}
		CwiseOpscalar(const Scalar& s,const other& o):base(0,o),s(s){}
		CwiseOpscalar(const CwiseOpscalar& another):base(nullptr,another._rhs),s(another.s){}
#ifdef DDA_SIMD
		inline __m128 coffe(std::size_t idx) const {
			if constexpr (rXpr)
				return _mm_mul_ps(_mm_set1_ps(s), this->_rhs->coffe(idx));
			else
				return _mm_mul_ps(_mm_set1_ps(s), _mm_load_ps(&this->_rhs->coffe(idx)));
		}
#else
		inline Scalar& coffe(std::size_t idx) const {
			return this->_rhs->coffe(idx)*s;
		}
#endif
	};

	template<typename Derived, typename otherDerived>
	CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator +(const Derived& l, const otherDerived& r) {
		return typename CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpsum(l,r);
	}

	template<typename Derived, typename otherDerived, typename std::enable_if<
														!std::is_arithmetic_v<Derived>&&!std::is_arithmetic_v<otherDerived>,int>::type=0>
	CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator *(const Derived& l, const otherDerived& r) {
		return typename CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpproduct(l, r);
	}

	template<typename Scalar,typename otherDerived,typename std::enable_if<
														std::is_arithmetic_v<Scalar>&&!std::is_arithmetic_v<otherDerived>,int>::type=0>
	CwiseOpscalar<Scalar,otherDerived> operator *(const Scalar& s,const otherDerived& o){
		return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
	}

	template<typename otherDerived, typename Scalar,typename std::enable_if<
														std::is_arithmetic_v<Scalar> && !std::is_arithmetic_v<otherDerived>,int>::type=0>
	CwiseOpscalar<Scalar, otherDerived> operator *(const otherDerived& o, const Scalar& s) {
		return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
	}
}