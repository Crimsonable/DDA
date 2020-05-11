#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"
#include "Expression.h"
#include "OpRegister.h"
#include "Functions.h"
#include "Assignment.h"
#include "MatrixMap.h"

namespace CSM {

	namespace internal {
		template<typename Scalar, typename Implment>
		struct traits<Matrix<Scalar, -1, -1, Implment>> {
			using scalar = Scalar;
			using Imp = Implment;
		};
	}


	template<typename Scalar, int Rows, int Cols, typename Implment=typename Functions::DefaultImp>
	class Matrix : public MatrixBase<Matrix<Scalar, Rows, Cols, Implment>>, public DenseStorage<Scalar>,
					public ExpBase<Matrix<Scalar,Rows,Cols, Implment>>{
	public:
		using scalar = Scalar;
		using Self = Matrix<Scalar, Rows, Cols, Implment>;
		typedef DenseStorage<Scalar> Storagebase;
		using Imp = Implment;
		static constexpr typename internal::OpRegister Op = OpRegister::None;
		bool defaultLazyAssign = false;
	public:
		Matrix(){}

		inline Storagebase& toStorage() {
			return *static_cast<Storagebase*>(this);
		}

		template<typename otherType>
		inline void operator=(const otherType& other) {
			AssignBase::template Assign<Self, otherType, Self>(this, std::forward<otherType>(*const_cast<otherType*>(&other)));
			defaultLazyAssign = false;
		}

		inline const scalar& coeff(std::size_t idx) const {
			return *(this->cdata() + idx);
		}

		inline scalar& coeffRef(std::size_t idx) {
			return *(this->data() + idx);
		}

		inline scalar& coeffRef(std::size_t r, std::size_t c) {
			return coeffRef(r + c * this->rows);
		}

		MatrixMap<Self> topLBottomR(const Index& topLeft, const Index& bottomRight) {
			size_t offset = topLeft.col*this->ld + topLeft.row;
			return MatrixMap<Self>(this->data() + offset, this->ld, bottomRight.row - topLeft.row + 1, bottomRight.col - topLeft.col + 1);
		}

		/*typename Block<Matrix<scalar,Rows,Cols>> topLeftBottomRight(Index&& topLeft, Index&& bottomRight) {
			return Block<Matrix<scalar, Rows, Cols>>(this, std::forward<Index>(topLeft), std::forward<Index>(bottomRight));
		}

		typename Block<Matrix<scalar, Rows, Cols>> row_block(Index&& left, Index&& right) {
			return Block<Matrix<scalar, Rows, Cols>>(this, std::forward<Index>(left), std::forward<Index>(right));
		}

		typename Block<Matrix<scalar, Rows, Cols>> col_block(Index&& top, Index&& down) {
			return Block<Matrix<scalar, Rows, Cols>>(this, std::forward<Index>(top), std::forward<Index>(down));
		}*/
	};
}
