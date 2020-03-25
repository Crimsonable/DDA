#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"
#include "Block.h"

namespace DDA {

	namespace internal {
		template<typename Scalar, int Rows, int Cols>
		struct traits<Matrix<Scalar, Rows, Cols>> {
			static constexpr int size = Rows * Cols;
			using scalar = Scalar;
			static constexpr bool isXpr = false;
			static constexpr bool isDot = false;
		}; 

		template<typename Scalar>
		struct traits<Matrix<Scalar, -1, -1>> {
			static constexpr int size = -1;
			using scalar = Scalar;
			static constexpr bool isXpr = false;
			static constexpr bool isDot = false;
		};
	}
	
	template<typename Scalar, int Rows, int Cols>
	class Matrix : public MatrixBase<Matrix<Scalar, Rows, Cols>>, public DenseStorage<Scalar, internal::traits<Matrix<Scalar, Rows, Cols>>::size , Rows, Cols>{
	public:
		using scalar = Scalar;
		typedef MatrixBase<Matrix<Scalar, Rows, Cols>> Matbase;
		typedef DenseStorage<Scalar, internal::traits<Matrix<Scalar, Rows, Cols>>::size, Rows, Cols> Storagebase;
		using traits = internal::traits<Matrix<Scalar, Rows, Cols>>;
	public:
		Matrix(){}
		Matrix(scalar* _array):Storagebase(_array){}
		Matrix(scalar* _array,int _size) :Storagebase(_array,_size) {}
		Matrix(std::initializer_list<scalar> _l):Storagebase(_l){}
		explicit Matrix(const Matrix& other):Storagebase(other.dataptr()){}
		Matrix(Matrix&& other):Storagebase(other.dataptr()){}

		inline Storagebase& toStorage() {
			return *static_cast<Storagebase*>(this);
		}

		inline void operator=(const Matrix& other) {
			static_cast<Matbase*>(this)->operator=(other);
		}

		template<typename otherDerived>
		inline void operator=(const otherDerived& other) {
			static_cast<Matbase*>(this)->operator=(other);
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

		typename Block<Matrix<scalar,Rows,Cols>> topLeftBottomRight(Index&& topLeft, Index&& bottomRight) {
			return Block<Matrix<scalar, Rows, Cols>>(*this, std::forward<Index>(topLeft), std::forward<Index>(bottomRight));
		}
	};
}
