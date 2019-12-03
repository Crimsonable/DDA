#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"

namespace DDA {

	namespace internal {
		template<typename Scalar, int Rows, int Cols>
		struct traits<Matrix<Scalar, Rows, Cols>> {
			static constexpr int size = Rows * Cols;
			using scalar = Scalar;
			static constexpr bool isXpr = false;
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
		Matrix(std::initializer_list<scalar> _l):Storagebase(_l){}
		explicit Matrix(const Matrix& other):Storagebase(other.dataptr()){}
		Matrix(Matrix&& other):Storagebase(other.dataptr()){}

		void operator=(Matrix& other) {
			static_cast<Matbase*>(this)->operator=(other);
		}

		template<typename otherDerived>
		void operator=(const otherDerived& other) {
			static_cast<Matbase*>(this)->operator=(other);
		}

		const scalar& coffe(std::size_t idx) const {
			return *(this->m_storage.array + idx);
		}

		scalar& coffeRef(std::size_t idx) {
			return *(this->m_storage.array + idx);
		}

		scalar& coffeRef(std::size_t r, std::size_t c) {
			return coffeRef(r + c * this->cols);
		}
	};
}
