#pragma once
#include "forwardDecleration.h"
#include "OpRegister.h"
#include "DenseStorage.h"

using namespace CSM::internal;
using std::size_t;


namespace CSM {

	namespace internal {
		template<typename Lhs,typename Rhs,typename Implment>
		struct traits<MatMulOp<Lhs, Rhs, Implment>> {
			using scalar = typename internal::traits<std::remove_reference_t<Lhs>>::scalar;
			using Imp = Implment;
		};

		template<typename Lhs, typename Rhs, typename Implment>
		struct traits<MatAddOp<Lhs, Rhs, Implment>> {
			using scalar = typename internal::traits<std::remove_reference_t<Lhs>>::scalar;
			using Imp = Implment;
		};

		template<typename Lhs, typename Rhs, typename Implment>
		struct traits<MatSubOp<Lhs, Rhs, Implment>> {
			using scalar = typename internal::traits<std::remove_reference_t<Lhs>>::scalar;
			using Imp = Implment;
		};

		template<typename Lhs, typename Rhs, typename Implment>
		struct traits<CwiseMulOp<Lhs, Rhs, Implment>> {
			using scalar = typename internal::traits<std::remove_reference_t<Lhs>>::scalar;
			using Imp = Implment;
		};

		template<typename Self, typename Implment>
		struct traits<TransposeOp<Self, Implment>> {
			using scalar = typename internal::traits<std::remove_reference_t<Self>>::scalar;
			using Imp = Implment;
		};
	}

	template<typename Derived>
	class ExpBase {
	private:
		FORCE_INLINE Derived* derived_exp() {return static_cast<Derived*>(this);}
	protected:
		using Implment = typename internal::traits<Derived>::Imp;
	public:
		Derived toExpression() {
			if constexpr (Derived::Op != OpRegister::None) {
				derived_exp()->setAllocFlag();
			}
			return *derived_exp();
		}

		template<typename other>
		MatMulOp<Derived, other, Implment> operator*(const other& r) {
			auto ptr = new MatMulOp<Derived, other, Implment>(static_cast<Derived*>(this), &r, static_cast<Derived*>(this)->rows, r.cols);
			return *ptr;
		}

		template<typename other>
		MatAddOp<Derived, other, Implment> operator+(const other& r) {
			auto ptr = new MatAddOp<Derived, other, Implment>(static_cast<Derived*>(this), &r, static_cast<Derived*>(this)->rows, r.cols);
			return *ptr;
		}

		template<typename other>
		MatSubOp<Derived, other, Implment> operator-(const other& r) {
			auto ptr = new MatSubOp<Derived, other, Implment>(static_cast<Derived*>(this), &r, static_cast<Derived*>(this)->rows, r.cols);
			return *ptr;
		}

		TransposeOp<Derived, Implment> transpose() {
			auto ptr = new TransposeOp<Derived, Implment>(static_cast<Derived*>(this), static_cast<Derived*>(this)->cols, static_cast<Derived*>(this)->rows);
			return *ptr;
		}
	};

	template<typename Lhs, typename Rhs>
	struct BinaryOpDispatcher :public CommonBinaryExpBase{
		mutable Lhs* l;
		mutable Rhs* r;
		bool deAllocate = true;
		size_t rows;
		size_t cols;
		using lhs_type = Lhs;
		using rhs_type = Rhs;

		BinaryOpDispatcher(const Lhs* l, const Rhs* r, size_t rows, size_t cols) :l(const_cast<Lhs*>(l)), r(const_cast<Rhs*>(r)),rows(rows),cols(cols) {}

		template<typename Function, typename ...Args>
		void disptach(Args ...args) {
			Function::apply(args...);
		}

		void setAllocFlag() {
			deAllocate = false;
			if constexpr (Lhs::Op != OpRegister::None)
				l->setAllocFlag();
			if constexpr (Rhs::Op != OpRegister::None)
				r->setAllocFlag();
		}
	};

	template<typename OpPtr>
	struct SingleOpDispatcher:public CommonSingleExpBase {
		mutable OpPtr* self;
		bool deAllocate = true;
		size_t rows, cols;
		using Type = OpPtr;

		SingleOpDispatcher(const OpPtr* p, size_t rows, size_t cols):self(const_cast<OpPtr*>(p)),rows(rows),cols(cols){}

		template<typename Function,typename ...Args>
		void dispatch(Args... args) {
			Function::apply(args...);
		}

		void setAllocFlag() {
			deAllocate = false;
			if constexpr (OpPtr::Op != OpRegister::None)
				self->setAllocFlag();
		}
	};

	template<typename Lhs, typename Rhs, typename Implment>
	class MatMulOp :public ExpBase<MatMulOp<Lhs, Rhs, Implment>>, public BinaryOpDispatcher<Lhs, Rhs> {
	public:
		using BinaryOpDispatcher<Lhs, Rhs>::BinaryOpDispatcher;
		void _clear() {this->~MatMulOp();}
		void clear() {
			if constexpr (Lhs::Op != OpRegister::None) this->l->clear();
			if constexpr (Rhs::Op != OpRegister::None) this->r->clear();
			_clear();
		}

		template<typename Dst,typename BasicBufferType>
		FORCE_INLINE void Eval(Dst* dst) {
			scalar *_lhs, *_rhs, *_dst;
			size_t lda, ldb;
			BasicBufferType *lhs_buffer = nullptr, *rhs_buffer = nullptr;
			if constexpr (Lhs::Op != OpRegister::None) {
				lhs_buffer = new BasicBufferType;
				lhs_buffer->alias() = *(this->l);
				_lhs = lhs_buffer->data();
				lda = lhs_buffer->ld;
			}
			else {
				_lhs = this->l->data();
				lda = this->l->ld;
			}
			if constexpr (Rhs::Op != OpRegister::None) {
				rhs_buffer = new BasicBufferType;
				rhs_buffer->alias() = *(this->r);
				_rhs = rhs_buffer->data();
				ldb = rhs_buffer->ld;
			}
			else {
				_rhs = this->r->data();
				ldb = this->r->ld;
			}
			this->template disptach<typename Implment::MatMulOp>(_lhs, lda, _rhs, ldb, dst->data(), dst->ld, this->rows, this->cols, this->l->cols);
			if (lhs_buffer) delete lhs_buffer;
			if (rhs_buffer) delete rhs_buffer;
		}

		using scalar = typename Lhs::scalar;
		static constexpr internal::OpRegister Op = OpRegister::MatMul;
		static constexpr bool defaultLazyAssign = false;
	};

	template<typename Lhs, typename Rhs, typename Implment, typename Function>
	class CwiseBinaryBase : public BinaryOpDispatcher<Lhs, Rhs> {
	public:
		using BinaryOpDispatcher<Lhs, Rhs>::BinaryOpDispatcher;
		
		template<typename Dst, typename BasicBufferType>
		FORCE_INLINE void Eval(Dst* dst) {
			scalar *_lhs, *_rhs, *_dst;
			size_t lda, ldb;
			BasicBufferType *lhs_buffer = nullptr;
			if constexpr (Lhs::Op != OpRegister::None) {
				lhs_buffer = new BasicBufferType;
				lhs_buffer->alias() = *(this->l);
				_lhs = lhs_buffer->data();
				lda = lhs_buffer->ld;
			}
			else {
				_lhs = this->l->data();
				lda = this->l->ld;
			}
			if constexpr (Rhs::Op != OpRegister::None) {
				dst->alias() = *(this->r);
				_rhs = dst->data();
				ldb = dst->ld;
			}
			else {
				_rhs = this->r->data();
				ldb = this->r->ld;
			}
			this->template disptach<Function>(_lhs, lda, _rhs, ldb, dst->data(), dst->ld, this->rows, this->cols);
			if (lhs_buffer) delete lhs_buffer;
		}

		using scalar = typename Lhs::scalar;
	};

	template<typename Lhs,typename Rhs,typename Implment>
	class MatAddOp :public ExpBase<MatAddOp<Lhs, Rhs, Implment>>, public CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::MatAddOp> {
	public:
		void _clear() { this->~MatAddOp(); }
		void clear() {
			if constexpr (Lhs::Op != OpRegister::None) this->l->clear();
			if constexpr (Rhs::Op != OpRegister::None) this->r->clear();
			_clear();
		}
		using CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::MatAddOp>::CwiseBinaryBase;
		static constexpr internal::OpRegister Op = OpRegister::MatAdd;
		static constexpr bool defaultLazyAssign = true;
	};

	template<typename Lhs, typename Rhs, typename Implment>
	class MatSubOp :public ExpBase<MatSubOp<Lhs, Rhs, Implment>>, public CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::MatSubOp> {
	public:
		void _clear() { this->~MatSubOp(); }
		void clear() {
			if constexpr (Lhs::Op != OpRegister::None) this->l->clear();
			if constexpr (Rhs::Op != OpRegister::None) this->r->clear();
			_clear();
		}
		using CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::MatSubOp>::CwiseBinaryBase;
		static constexpr internal::OpRegister Op = OpRegister::MatSub;
		static constexpr bool defaultLazyAssign = true;
	};

	template<typename Lhs, typename Rhs, typename Implment>
	class CwiseMulOp :public ExpBase<CwiseMulOp<Lhs, Rhs, Implment>>, public CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::CwiseMulOp> {
	public:
		void _clear() { this->~CwiseMulOp(); }
		void clear() {
			if constexpr (Lhs::Op != OpRegister::None) this->l->clear();
			if constexpr (Rhs::Op != OpRegister::None) this->r->clear();
			_clear();
		}
		using CwiseBinaryBase<Lhs, Rhs, Implment, typename Implment::CwiseMulOp>::CwiseBinaryBase;
		static constexpr internal::OpRegister Op = OpRegister::CwiseMul;
		static constexpr bool defaultLazyAssign = true;
	};

	template<typename Self,typename Implment, typename Function>
	class SingleOpBase :public SingleOpDispatcher<Self> {
	public:
		using SingleOpDispatcher<Self>::SingleOpDispatcher;
		void clear() { this->~SingleOpBase(); }

		template<typename Dst,typename BasicBufferType>
		FORCE_INLINE void Eval(Dst* dst) {
			BasicBufferType *buffer = nullptr;
			scalar *data_ptr;
			size_t ld;
			if constexpr (Self::Op != OpRegister::None) {
				buffer = new BasicBufferType;
				buffer->alias() = *(this->self);
				data_ptr = buffer->data();
				ld = buffer->ld;
			}
			else {
				data_ptr = this->self->data();
				ld = this->self->ld;
			}
			this->template dispatch<Function>(data_ptr, ld, dst->data(), dst->ld, this->cols, this->rows);
		}

		using scalar = typename Self::scalar;
	};

	template<typename Self,typename Implment>
	class TransposeOp :public ExpBase<TransposeOp<Self, Implment>>, public SingleOpBase<Self, Implment, typename Implment::TransposeOp> {
	public:
		void _clear() { this->~TransposeOp(); }
		void clear() {
			if constexpr (Self::Op != OpRegister::None) this->self->clear();
			_clear();
		}

		using SingleOpBase<Self, Implment, typename Implment::TransposeOp>::SingleOpBase;
		using scalar = typename Self::scalar;
		static constexpr internal::OpRegister Op = OpRegister::Transpose;
		static constexpr bool defaultLazyAssign = false;
	};
}