#pragma once
#include "Matrix.h"
#include "Transpose.h"
#include "forwardDecleration.h"
#include "SIMD.h"

namespace DDA {
	namespace internal {
		template<typename Self>
		struct traits<TransOp<Self>> {
			static constexpr int size = traits<Self>::size;
			using scalar = typename traits<Self>::scalar;
			static constexpr bool isXpr = true;
			static constexpr bool isDot = false;
		};
	}

	template <typename Self>
	class SingleOp {
	private:
		using dtype = typename internal::traits<Self>::scalar;
	public:
		Self* self;
		SingleOp(const Self& _self) { self = const_cast<Self*>(&_self); }

		SingleOp(const Self* _self) { self = const_cast<Self*>(_self); }

		template <typename src, typename dst>
		void run(dst* d, const src& s)
		{
			src* temp_ptr = const_cast<src*>(&s);
			temp_ptr->template RecurFun<v_128<dtype>>(0);
			d->share(temp_ptr->temp_res_ptr);
		}
	};

	template <typename Self>
	class TransOp : public SingleOp<Self> {
	private:
		typedef SingleOp<Self> base;
		using scalar = typename internal::traits<Self>::scalar;
		using dynamicMat = Matrix<scalar, -1, -1>;
		using dynamicMat_ptr = std::shared_ptr<dynamicMat>;
		static constexpr bool isXpr = internal::traits<Self>::isXpr;
		static constexpr bool isXprDot = internal::traits<Self>::isDot;

	public:
		dynamicMat_ptr temp_res_ptr, temp_exp_ptr;
		int rows = this->self->rows;
		int cols = this->self->cols;

		TransOp(){}
		TransOp(const Self& _self):base(_self){}

		template<typename Vtype,typename vtype=typename internal::traits<Vtype>::vtype>
		inline vtype RecurFun(std::size_t idx)
		{
			vtype ret_val;
			if (!temp_res_ptr.get())
			{
				temp_res_ptr = std::make_shared<dynamicMat>();
				temp_res_ptr->resize(cols, rows);
				temp_exp_ptr = std::make_shared<dynamicMat>();
				if constexpr (isXpr)
				{
					*temp_exp_ptr = *this->self;
				}
				else 
				{
					temp_exp_ptr->share(this->self);
				}
			}
			transpose(temp_exp_ptr.get(), temp_res_ptr.get());
			load_ps(ret_val, &temp_res_ptr->coeffRef(idx));
			return ret_val;
		}

		base &toXprBase() {
			return *static_cast<base*>(this);
		}
	};

	template<typename Self>
	TransOp<Self> Transpose(const Self& s) {
		return TransOp<Self>(s);
	}

} // namespace DDA
