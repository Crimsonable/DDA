#pragma once
#include "MatrixBase.h"
#include "Matrix.h"


namespace DDA {
	namespace internal {
		template<typename Scalar,typename lhs,typename rhs>
		struct traits<CwiseOpsum<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
			static constexpr bool isProduct = false;
		};

		template<typename Scalar,typename lhs,typename rhs>
		struct traits<CwiseOpproduct<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
			static constexpr bool isProduct = false;
		};

		template<typename Scalar,typename other>
		struct traits<CwiseOpscalar<Scalar, other>> {
			static constexpr int size = traits<other>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
			static constexpr bool isProduct = false;
		};

		template<typename Scalar,typename lhs,typename rhs>
		struct traits<ProductOp<Scalar, lhs, rhs>> {
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			static constexpr bool isXpr = true;
			static constexpr bool isProduct = true;
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

		template<typename src, typename dst>
		void run(dst* d, const src& s) {
			d->resize(s.rows, s.cols);
			int steps = d->size % 4 ? d->size / 4 + 1 : d->size / 4;
			int strides = 4;
			int SimdSize = steps * strides;
			auto dstptr = d->data();
			for (int i = 0; i < SimdSize; i += strides) {
				_mm_store_ps(dstptr, s.coeff(i));
				dstptr += strides;
			}
		}
	};

	template<typename Scalar, typename lhs,typename rhs>
	class CwiseOpsum: public BinaryOp<lhs,rhs>{
	public:
		typedef BinaryOp<lhs, rhs> base;
		static constexpr bool lXpr = internal::traits<lhs>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs>::isXpr;
		int cols = this->_lhs->cols;
		int rows = this->_rhs->rows;
		CwiseOpsum(){}
		CwiseOpsum(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpsum(const CwiseOpsum& other):base(other._lhs,other._rhs) {}
#ifdef DDA_SIMD
		inline __m128 coeff(std::size_t idx) const {
			if constexpr (lXpr || rXpr) {
				if constexpr (lXpr && rXpr)
					return _mm_add_ps(this->_lhs->coeff(idx), this->_rhs->coeff(idx));
				else if constexpr (lXpr && !rXpr)
					return _mm_add_ps(this->_lhs->coeff(idx), _mm_load_ps(&this->_rhs->coeff(idx)));
				else
					return _mm_add_ps(_mm_load_ps(&this->_lhs->coeff(idx)), this->_rhs->coeff(idx));
			}
			else
				return _mm_add_ps(_mm_load_ps(&this->_lhs->coeff(idx)), _mm_load_ps(&this->_rhs->coeff(idx)));
		}

#else

		inline const Scalar& coeff(std::size_t idx) const {
			return this->_lhs->coeff(idx) + this->_rhs->coeff(idx);
		}
#endif

		base& toXprBase() {
			return *static_cast<base*>(this);
		}

	};

	template<typename Scalar,typename lhs,typename rhs>
	class CwiseOpproduct : public BinaryOp<lhs, rhs> {
	public:
		typedef BinaryOp<lhs, rhs> base;
		static constexpr bool lXpr = internal::traits<lhs>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs>::isXpr;
		int cols = this->_lhs->cols;
		int rows = this->_rhs->rows;
		CwiseOpproduct(){}
		CwiseOpproduct(const lhs& l, const rhs& r):base(l,r){}
		CwiseOpproduct(const CwiseOpproduct& other):base(other._lhs,other._rhs){}
#ifdef DDA_SIMD
		inline __m128 coeff(std::size_t idx) const {
			if constexpr (lXpr || rXpr) {
				if constexpr (lXpr && rXpr)
					return _mm_mul_ps(this->_lhs->coeff(idx), this->_rhs->coeff(idx));
				else if constexpr (lXpr && !rXpr)
					return _mm_mul_ps(this->_lhs->coeff(idx), _mm_load_ps(&this->_rhs->coeff(idx)));
				else
					return _mm_mul_ps(_mm_load_ps(&this->_lhs->coeff(idx)), this->_rhs->coeff(idx));
			}
			else
				return _mm_mul_ps(_mm_load_ps(&this->_lhs->coeff(idx)), _mm_load_ps(&this->_rhs->coeff(idx)));
		}
#else
		inline const Scalar& coeff(std::size_t idx) const {
			return this->_lhs->coeff(idx) * this->_rhs->coeff(idx);
		}
#endif

		base& toXprBase() {
			return *static_cast<base*>(this);
		}
	};

	template<typename Scalar,typename other>
	class CwiseOpscalar : public BinaryOp<Scalar, other> {
	private:
		Scalar s;
	public:
		typedef BinaryOp<Scalar, other> base;
		static constexpr bool rXpr = internal::traits<other>::isXpr;
		int cols = this->_rhs->cols;
		int rows = this->_rhs->rows;
		CwiseOpscalar(){}
		CwiseOpscalar(const Scalar& s,const other& o):base(0,o),s(s){}
		CwiseOpscalar(const CwiseOpscalar& other):base(nullptr,other._rhs),s(other.s){}
#ifdef DDA_SIMD
		inline __m128 coeff(std::size_t idx) const {
			if constexpr (rXpr)
				return _mm_mul_ps(_mm_set1_ps(s), this->_rhs->coeff(idx));
			else
				return _mm_mul_ps(_mm_set1_ps(s), _mm_load_ps(&this->_rhs->coeff(idx)));
		}
#else
		inline Scalar& coeff(std::size_t idx) const {
			return this->_rhs->coeff(idx)*s;
		}
#endif

		base& toXprBase() {
			return *static_cast<base*>(this);
		}
	};

	template<typename Scalar, typename lhs,typename rhs>
	class ProductOp :public BinaryOp<lhs, rhs> {
	public:
		typedef BinaryOp<lhs, rhs> base;
		static constexpr bool lXpr = internal::traits<lhs>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs>::isXpr;
		int cols = this->_rhs->cols;
		int rows = this->_lhs->rows;
		ProductOp(){}
		ProductOp(const lhs& l,const rhs& r):base(l,r){}
		ProductOp(const ProductOp& other):base(other._lhs,other._rhs){}

		inline void coeff(std::size_t i, Scalar* dst) {
			Matrix<Scalar, -1, -1, -1> left, right;
		}
	};


	template<typename Derived, typename otherDerived>
	CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator +(const Derived& l, const otherDerived& r) {
		return typename CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpsum(l, r);
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