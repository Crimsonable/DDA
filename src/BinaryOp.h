#pragma once
#include "MatrixBase.h"
#include "Matrix.h"
#include "SIMD.h"
#include "Product.h"

namespace DDA
{
namespace internal
{
template <typename Scalar, typename lhs, typename rhs>
struct traits<CwiseOpsum<Scalar, lhs, rhs>>
{
    static constexpr int size = traits<lhs>::size;
    using scalar = Scalar;
    static constexpr bool isXpr = true;
    static constexpr bool isDot = false;
};

template <typename Scalar, typename lhs, typename rhs>
struct traits<CwiseOpproduct<Scalar, lhs, rhs>>
{
    static constexpr int size = traits<lhs>::size;
    using scalar = Scalar;
    static constexpr bool isXpr = true;
    static constexpr bool isDot = false;
};

template <typename Scalar, typename other>
struct traits<CwiseOpscalar<Scalar, other>>
{
    static constexpr int size = traits<other>::size;
    using scalar = Scalar;
    static constexpr bool isXpr = true;
    static constexpr bool isDot = false;
};

template <typename Scalar, typename lhs, typename rhs>
struct traits<MatDotOp<Scalar, lhs, rhs>>
{
    static constexpr int size = traits<lhs>::size;
    using scalar = Scalar;
    static constexpr bool isXpr = true;
    static constexpr bool isDot = true;
};
} // namespace internal

template <typename lhs, typename rhs>
class BinaryOp
{
    using dtype = typename internal::traits<rhs>::scalar;

public:
    lhs *_lhs;
    rhs *_rhs;
    BinaryOp(const lhs &l, const rhs &r)
    {
        _lhs = const_cast<lhs *>(&l);
        _rhs = const_cast<rhs *>(&r);
    }
    BinaryOp(const lhs *l, const rhs *r)
    {
        _lhs = const_cast<lhs *>(l);
        _rhs = const_cast<rhs *>(r);
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
            store(dstptr, s.template RecurFun<v_256<dtype>>(i));
            dstptr += main_step;
        }
        int after_pad = half_step + d->size - (d->size - d->size % main_step) % half_step;
        for (int i = EndVec; i < after_pad; i += half_step)
        {
            store(dstptr, s.template RecurFun<v_128<dtype>>(i));
            dstptr += half_step;
        }
    }

    template <typename src, typename dst, typename std::enable_if<internal::traits<src>::isDot, int>::type = 0>
    void run(dst *d, const src &s)
    {
        src *temp_src = const_cast<src *>(&s);
        temp_src->template RecurFun<v_128<dtype>>(0);
        d->swap(temp_src->tempDotRes);
    }
};

template <typename Scalar, typename lhs, typename rhs>
class CwiseOpsum : public BinaryOp<lhs, rhs>
{
public:
    typedef BinaryOp<lhs, rhs> base;
    static constexpr bool lXpr = internal::traits<lhs>::isXpr;
    static constexpr bool rXpr = internal::traits<rhs>::isXpr;
    static constexpr bool islXprDot = internal::traits<lhs>::isDot;
    static constexpr bool isrXprDot = internal::traits<rhs>::isDot;
    int cols = this->_lhs->cols;
    int rows = this->_rhs->rows;
    CwiseOpsum() {}
    CwiseOpsum(const lhs &l, const rhs &r) : base(l, r) {}
    CwiseOpsum(const CwiseOpsum &other) : base(other._lhs, other._rhs) {}
#ifdef SIMD
    template <typename Vtype>
    inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx) const
    {
        typename internal::traits<Vtype>::vtype l, r;
        if constexpr (lXpr || rXpr)
        {
            if constexpr (lXpr && rXpr)
                return this->_lhs->template RecurFun<Vtype>(idx) + this->_rhs->template RecurFun<Vtype>(idx);
            else if constexpr (lXpr && !rXpr)
            {
                load_ps(r, &this->_rhs->coeff(idx));
                return this->_lhs->template RecurFun<Vtype>(idx) + r;
            }
            else
            {
                load_ps(l, &this->_lhs->coeff(idx));
                return l + this->_rhs->template RecurFun<Vtype>(idx);
            }
        }
        else
        {
            load_ps(l, &this->_lhs->coeff(idx));
            load_ps(r, &this->_rhs->coeff(idx));
            return l + r;
        }
    }

#else

    inline const Scalar &coeff(std::size_t idx) const
    {
        return this->_lhs->coeff(idx) + this->_rhs->coeff(idx);
    }
#endif

    base &toXprBase()
    {
        return *static_cast<base *>(this);
    }
};

template <typename Scalar, typename lhs, typename rhs>
class CwiseOpproduct : public BinaryOp<lhs, rhs>
{
public:
    typedef BinaryOp<lhs, rhs> base;
    static constexpr bool lXpr = internal::traits<lhs>::isXpr;
    static constexpr bool rXpr = internal::traits<rhs>::isXpr;
    static constexpr bool islXprDot = internal::traits<lhs>::isDot;
    static constexpr bool isrXprDot = internal::traits<rhs>::isDot;
    int cols = this->_lhs->cols;
    int rows = this->_rhs->rows;
    CwiseOpproduct() {}
    CwiseOpproduct(const lhs &l, const rhs &r) : base(l, r) {}
    CwiseOpproduct(const CwiseOpproduct &other) : base(other._lhs, other._rhs) {}
#ifdef SIMD
    template <typename Vtype>
    inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx) const
    {
        typename internal::traits<Vtype>::vtype l, r;
        if constexpr (lXpr || rXpr)
        {
            if constexpr (lXpr && rXpr)
                return this->_lhs->template RecurFun<Vtype>(idx) * this->_rhs->template RecurFun<Vtype>(idx);
            else if constexpr (lXpr && !rXpr)
            {
                load_ps(r, &this->_rhs->coeff(idx));
                return this->_lhs->template RecurFun<Vtype>(idx) * r;
            }
            else
            {
                load_ps(l, &this->_lhs->coeff(idx));
                return l * this->_rhs->template RecurFun<Vtype>(idx);
            }
        }
        else
        {
            load_ps(l, &this->_lhs->coeff(idx));
            load_ps(r, &this->_rhs->coeff(idx));
            return l * r;
        }
    }
#else
    inline const Scalar &coeff(std::size_t idx) const
    {
        return this->_lhs->coeff(idx) * this->_rhs->coeff(idx);
    }
#endif

    base &toXprBase()
    {
        return *static_cast<base *>(this);
    }
};

template <typename Scalar, typename other>
class CwiseOpscalar : public BinaryOp<Scalar, other>
{
private:
    Scalar s;

public:
    typedef BinaryOp<Scalar, other> base;
    static constexpr bool rXpr = internal::traits<other>::isXpr;
    static constexpr bool isrXprDot = internal::traits<other>::isDot;
    int cols = this->_rhs->cols;
    int rows = this->_rhs->rows;
    CwiseOpscalar() {}
    CwiseOpscalar(const Scalar &s, const other &o) : base(0, o), s(s) {}
    CwiseOpscalar(const CwiseOpscalar &another) : base(nullptr, another._rhs), s(another.s) {}
#ifdef SIMD
    template <typename Vtype>
    inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx) const
    {
        typename internal::traits<Vtype>::vtype l, r;
        load_ps1(l, &s);
        if constexpr (rXpr)
            return l * this->_rhs->template RecurFun<Vtype>(idx);
        else
        {
            load_ps(r, &this->_rhs->coeff(idx));
            return l * r;
        }
    }
#else
    inline Scalar &coeff(std::size_t idx) const
    {
        return this->_rhs->coeff(idx) * s;
    }
#endif

    base &toXprBase()
    {
        return *static_cast<base *>(this);
    }
};

template <typename Scalar, typename lhs, typename rhs>
class MatDotOp : public BinaryOp<lhs, rhs>
{
private:
    using dynamicMat = Matrix<Scalar, -1, -1>;
    using dynamicMat_ptr = std::shared_ptr<dynamicMat>;
	typedef BinaryOp<lhs, rhs> base;

public:
    dynamicMat_ptr tempDotRes, tempLhsRes, tempRhsRes;
    static constexpr bool lXpr = internal::traits<lhs>::isXpr;
    static constexpr bool rXpr = internal::traits<rhs>::isXpr;
    static constexpr bool islXprDot = internal::traits<lhs>::isDot;
    static constexpr bool isrXprDot = internal::traits<rhs>::isDot;
    int rows = this->_lhs->rows;
    int cols = this->_rhs->cols;
    MatDotOp() {}
    MatDotOp(const lhs &l, const rhs &r) : base(l, r)
    {
        //tempDotRes.resize(rows, cols);
    }
    MatDotOp(const MatDotOp &other) : base(other._lhs, other._rhs) {}

    template <typename Vtype>
    inline typename internal::traits<Vtype>::vtype RecurFun(std::size_t idx)
    {
        typename internal::traits<Vtype>::vtype ret_val;
        if (!tempDotRes.get())
        {
            tempDotRes = std::make_shared<dynamicMat>();
            tempDotRes->resize(rows, cols);
            if constexpr (islXprDot)
            {
                this->_lhs->template RecurFun<Vtype>(idx);
                tempLhsRes->share(this->_lhs->tempDotRes);
            }
            else
            {
                tempLhsRes = std::make_shared<dynamicMat>();
                *tempLhsRes = *this->_lhs;
            }
            if constexpr (isrXprDot)
            {
                this->_rhs->template RecurFun<Vtype>(idx);
                tempRhsRes = this->_rhs->tempDotRes;
            }
            else
            {
                tempRhsRes = std::make_shared<dynamicMat>();
                *tempRhsRes = *this->_rhs;
            }
            Product(tempLhsRes.get(), tempRhsRes.get(), tempDotRes.get());
        }
        load_ps(ret_val, &tempDotRes->coeffRef(idx));
        return ret_val;
    }

    base &toXprBase()
    {
        return *static_cast<base *>(this);
    }
};

template <typename Derived, typename otherDerived>
CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator+(const Derived &l, const otherDerived &r)
{
    return typename CwiseOpsum<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpsum(l, r);
}

template <typename Derived, typename otherDerived>
MatDotOp<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator*(const Derived &l, const otherDerived &r)
{
    return typename MatDotOp<typename internal::traits<Derived>::scalar, Derived, otherDerived>::MatDotOp(l, r);
}

/*template<typename Derived, typename otherDerived, typename std::enable_if<
														!std::is_arithmetic_v<Derived>&&!std::is_arithmetic_v<otherDerived>,int>::type=0>
	CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived> operator *(const Derived& l, const otherDerived& r) {
		return typename CwiseOpproduct<typename internal::traits<Derived>::scalar, Derived, otherDerived>::CwiseOpproduct(l, r);
	}*/

template <typename Scalar, typename otherDerived, typename std::enable_if<std::is_arithmetic_v<Scalar> && !std::is_arithmetic_v<otherDerived>, int>::type = 0>
CwiseOpscalar<Scalar, otherDerived> operator*(const Scalar &s, const otherDerived &o)
{
    return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
}

template <typename otherDerived, typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar> && !std::is_arithmetic_v<otherDerived>, int>::type = 0>
CwiseOpscalar<Scalar, otherDerived> operator*(const otherDerived &o, const Scalar &s)
{
    return typename CwiseOpscalar<Scalar, otherDerived>::CwiseOpscalar(s, o);
}
} // namespace DDA