#pragma once
#include "MatrixBase.h"
#include "Matrix.h"
#include "SIMD.h"
#include "Functions.h"

namespace DDA
{
	namespace internal
	{
		template<>
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
		};

		template <typename Scalar, typename other>
		struct traits<CwiseOpscalar<Scalar, other>>
		{
			static constexpr int size = traits<other>::size;
			using scalar = Scalar;
			using lhs_type = Scalar;
			using rhs_type = other;
			static constexpr bool isXpr = true;
			static constexpr bool isDot = false;
		};

		template <typename Scalar, typename lhs, typename rhs>
		struct traits<MatDotOp<Scalar, lhs, rhs>>
		{
			static constexpr int size = traits<lhs>::size;
			using scalar = Scalar;
			using lhs_type = lhs;
			using rhs_type = rhs;
			static constexpr bool isXpr = true;
			static constexpr bool isDot = true;
		};
	} // namespace internal

	template <typename Derived>
	class BinaryOp
	{
		using lhs_type = typename internal::traits<Derived>::lhs_type;
		using rhs_type = typename internal::traits<Derived>::rhs_type;
		using dtype = typename internal::traits<rhs_type>::scalar;
		Derived* derived_ptr;
	public:
		static constexpr bool lXpr = internal::traits<lhs_type>::isXpr;
		static constexpr bool rXpr = internal::traits<rhs_type>::isXpr;
		static constexpr bool islXprDot = internal::traits<lhs_type>::isDot;
		static constexpr bool isrXprDot = internal::traits<rhs_type>::isDot;
		lhs_type *_lhs;
		rhs_type *_rhs;

		BinaryOp() {
			derived_ptr = derived();
		}

		~BinaryOp() {
			if constexpr (lXpr)
				_lhs->~lhs_type();
			if constexpr (rXpr)
				_rhs->~rhs_type();
		}

		Derived* derived() {
			return reinterpret_cast<Derived*>(this);
		}

		template <typename Vtype, typename Dst>
		inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx, Dst* d) const
		{
			typename internal::traits<Vtype>::vtype l, r;
			if constexpr (lXpr || rXpr)
			{
				if constexpr (lXpr && rXpr)
					return derived_ptr->template func<Vtype>(_lhs->template RecurFun<Vtype, Dst>(idx, d), _rhs->template RecurFun<Vtype, Dst>(idx, d));
				else if constexpr (lXpr && !rXpr)
				{
					load_ps(r, &this->_rhs->coeff(idx));
					return derived_ptr->template func<Vtype>(_lhs->template RecurFun<Vtype, Dst>(idx, d), r);
				}
				else
				{
					load_ps(l, &this->_lhs->coeff(idx));
					return derived_ptr->template func<Vtype>(l, _rhs->template RecurFun<Vtype, Dst>(idx, d));
				}
			}
			else
			{
				load_ps(l, &_lhs->coeff(idx));
				load_ps(r, &_rhs->coeff(idx));
				return derived_ptr->template func<Vtype>(l, r);
			}
		}

		template <typename src, typename dst, typename std::enable_if<!internal::traits<src>::isDot, int>::type = 0>
		void run(dst *d, const src &s)
		{
			d->resize(s.rows, s.cols);

			auto dstptr = d->data();
			constexpr int main_step = 256 / (sizeof(dtype) * 8);
			constexpr int half_step = main_step / 2;
			int EndVec = d->size - d->size % main_step;

			for (int i = 0; i < EndVec; i += main_step)
			{
				store(dstptr, s.template RecurFun<v_256<dtype>, dst>(i, d));
				dstptr += main_step;
			}
			int after_pad = half_step + d->size - (d->size - d->size % main_step) % half_step;
			for (int i = EndVec; i < after_pad; i += half_step)
			{
				store(dstptr, s.template RecurFun<v_128<dtype>, dst>(i, d));
				dstptr += half_step;
			}
		}

		template <typename src, typename dst, typename std::enable_if<internal::traits<src>::isDot, int>::type = 0>
		void run(dst *d, const src &s)
		{
			src *temp_src = const_cast<src *>(&s);
			temp_src->template RecurFun<v_128<dtype>, dst>(0, d);
			d->share(temp_src->tempDotRes);
		}
	};

	template <typename Scalar, typename lhs, typename rhs>
	class CwiseOpsum : public BinaryOp<CwiseOpsum<Scalar, lhs, rhs>>
	{
	public:
		typedef BinaryOp<CwiseOpsum<Scalar, lhs, rhs>> base;
		int rows, cols;
		CwiseOpsum() {}
		CwiseOpsum(const lhs &l, const rhs &r) :base() {
			this->_lhs = const_cast<lhs*>(&l);
			this->_rhs = const_cast<rhs*>(&r);
			rows = this->_lhs->rows;
			cols = this->_rhs->cols;
		}

		template<typename Vtype, typename SIMD_INS_TYPE = typename internal::traits<Vtype>::vtype>
		inline SIMD_INS_TYPE func(const SIMD_INS_TYPE& l, const SIMD_INS_TYPE& r) {
			return l + r;
		}

		base &toXprBase()
		{
			return *static_cast<base*>(this);
		}
	};

	template <typename Scalar, typename lhs, typename rhs>
	class CwiseOpproduct : public BinaryOp<CwiseOpproduct<Scalar, lhs, rhs>>
	{
	public:
		typedef BinaryOp<CwiseOpproduct<Scalar, lhs, rhs>> base;
		int rows, cols;
		CwiseOpproduct() {}
		CwiseOpproduct(const lhs &l, const rhs &r) :base() {
			this->_lhs = const_cast<lhs*>(&l);
			this->_rhs = const_cast<rhs*>(&r);
			rows = this->_lhs->rows;
			cols = this->_rhs->cols;
		}

		template<typename Vtype, typename SIMD_INS_TYPE = typename internal::traits<Vtype>::vtype>
		inline SIMD_INS_TYPE func(const SIMD_INS_TYPE& l, const SIMD_INS_TYPE& r) {
			return l * r;
		}

		base &toXprBase()
		{
			return *static_cast<base *>(this);
		}
	};

	template <typename Scalar, typename other>
	class CwiseOpscalar : public BinaryOp<CwiseOpscalar<Scalar, other>>
	{
	private:
		Scalar s;
	public:
		typedef BinaryOp<CwiseOpscalar<Scalar, other>> base;
		int rows, cols;
		CwiseOpscalar() {}
		CwiseOpscalar(const Scalar &s, const other &o) :base() {
			this->_lhs = const_cast<Scalar*>(&s);
			this->_rhs = const_cast<other*>(&o);
			rows = this->_lhs->rows;
			cols = this->_rhs->cols;
		}

		template<typename Vtype, typename SIMD_INS_TYPE = typename internal::traits<Vtype>::vtype>
		inline SIMD_INS_TYPE func(const SIMD_INS_TYPE& l, const SIMD_INS_TYPE& r) {
			return l * r;
		}

		template <typename Vtype>
		inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx) const
		{
			typename internal::traits<Vtype>::vtype l, r;
			load_ps1(l, &s);
			if constexpr (this->rXpr)
				return func<Vtype>(l, this->_rhs->template RecurFun<Vtype>(idx));
			else
			{
				load_ps(r, &this->_rhs->coeff(idx));
				return func<Vtype>(l, r);
			}
		}

		base &toXprBase()
		{
			return *static_cast<base *>(this);
		}
	};

	template <typename Scalar, typename lhs, typename rhs>
	class MatDotOp : public BinaryOp<MatDotOp<Scalar, lhs, rhs>>
	{
	private:
		using dynamicMat = Matrix<Scalar, -1, -1>;
		typedef BinaryOp<MatDotOp<Scalar, lhs, rhs>> base;
	public:
		dynamicMat *tempDotRes = nullptr, *tempLhsRes = nullptr, *tempRhsRes = nullptr;
		int rows, cols;

		MatDotOp() {}
		MatDotOp(const lhs &l, const rhs &r) :base()
		{
			this->_lhs = const_cast<lhs*>(&l);
			this->_rhs = const_cast<rhs*>(&r);
			rows = this->_lhs->rows;
			cols = this->_rhs->cols;
		}
		~MatDotOp() {
			delete tempDotRes;
		}

		template<typename Dst>
		FORCE_INLINE bool CHECK_SIZE(Dst* d) {
			return d->rows == rows && d->cols == cols;
		}

		template <typename Vtype, typename Dst>
		inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx, Dst* d)
		{
			typename internal::traits<Vtype>::vtype ret_val;
			if (!tempDotRes) {
				tempDotRes = new dynamicMat();
				if (CHECK_SIZE(d)) 
					tempDotRes->share(d);
				else
					tempDotRes->resize(rows, cols);
				if constexpr (this->lXpr) {
					tempLhsRes = new dynamicMat();
					*tempLhsRes = *this->_lhs;
				}
				else
					tempLhsRes = this->_lhs;
				if constexpr (this->rXpr) {
					tempRhsRes = new dynamicMat();
					*tempRhsRes = *this->_rhs;
				}
				else
					tempRhsRes = this->_rhs;
				Product(tempLhsRes, tempRhsRes, tempDotRes);
			}
			if constexpr (this->lXpr)
				delete tempLhsRes;
			if constexpr (this->rXpr)
				delete tempRhsRes;
			load_ps(ret_val, &tempDotRes->coeffRef(idx));
			return ret_val;
		}

		base &toXprBase()
		{
			return *static_cast<base *>(this);
		}
	};

	template <typename Derived, typename otherDerived, typename RetType=typename CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived>>
	RetType operator+(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::CwiseOpsum(l, r));
	}

	template <typename Derived, typename otherDerived, typename RetType= typename MatDotOp<typename internal::traits<Derived>::scalar, Derived, otherDerived>>
	RetType operator*(const Derived &l, const otherDerived &r)
	{
		return *(new typename RetType::MatDotOp(l, r));
	}

	template<typename Derived,
		typename otherDerived, 
		typename RetType=typename CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived>,
		typename std::enable_if<!std::is_arithmetic_v<Derived> && !std::is_arithmetic_v<otherDerived>, int>::type = 0>
		RetType CwiseSum(const Derived& l, const otherDerived& r) {
		return *(new typename RetType::CwiseOpproduct(l, r));
	}

	template <typename Scalar,
		typename otherDerived,
		typename RetType=typename CwiseOpscalar<Scalar, otherDerived>,
		typename std::enable_if<std::is_arithmetic_v<Scalar> && !std::is_arithmetic_v<otherDerived>, int>::type = 0>
	RetType operator*(const Scalar &s, const otherDerived &o)
	{
		return *(new typename RetType::CwiseOpscalar(s, o));
	}

	template <typename otherDerived,
		typename Scalar,
		typename RetType=typename CwiseOpscalar<Scalar, otherDerived>,
		typename std::enable_if<std::is_arithmetic_v<Scalar> && !std::is_arithmetic_v<otherDerived>, int>::type = 0>
	RetType operator*(const otherDerived &o, const Scalar &s)
	{
		return *(new typename RetType::CwiseOpscalar(s, o));
	}
} // namespace DDA