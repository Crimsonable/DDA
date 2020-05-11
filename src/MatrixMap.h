#pragma once
#include "DenseStorage.h"
#include "MatrixBase.h"
#include "Assignment.h"

namespace CSM {
	namespace internal {
		template<typename MatType>
		struct traits<MatrixMap<MatType>> {
			using scalar = typename internal::traits<MatType>::scalar;
			using Imp = typename internal::traits<MatType>::Imp;
		};
	}

	template<typename MatType>
	class MatrixMap :public MatrixBase<MatrixMap<MatType>>, public DenseStorageMap<typename internal::traits<MatType>::scalar>,public CommonMapBase {
		using Storage = DenseStorageMap<typename internal::traits<MatType>::scalar>;
		using Self = MatrixMap<MatType>;
	public:
		using scalar = typename internal::traits<MatType>::scalar;
		static constexpr typename internal::OpRegister Op = OpRegister::None;
		using Imp = typename internal::traits<MatType>::Imp;
		bool defaultLazyAssign = true;

		MatrixMap(scalar* data, int ld, int rows, int cols) :Storage(data, ld, rows, cols){}

		template<typename otherType>
		inline void operator=(const otherType& other) {
			AssignBase::template Assign<Self, otherType, MatType>(this, std::forward<otherType>(*const_cast<otherType*>(&other)));
		}

		inline const scalar& coeff(std::size_t idx) const {
			return *(this->cdata() + idx / this->rows*this->ld + idx % this->rows);
		}

		inline scalar& coeffRef(std::size_t idx) {
			return *(this->data() + idx / this->rows*this->ld + idx % this->rows);
		}

		inline scalar& coeffRef(int row, int col) {
			return coeffRef(row + col * this->rows);
		}
	};
}