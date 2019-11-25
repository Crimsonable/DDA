#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"

namespace DDA {

	namespace internal {
		template<class Scalar, int Rows, int Cols>
		struct traits<Matrix<Scalar, Rows, Cols>> {
			/*enum {
				size = (Rows == Dynamic || Cols == Dynamic) ? 0 : Rows * Cols,
			};*/
			static constexpr int size = Rows * Cols;
			using scalar = Scalar;
		}; 
	}
	

	template<class Scalar, int Rows, int Cols>
	class Matrix : public MatrixBase<Matrix<Scalar, Rows, Cols>> {
	public:
		using scalar = Scalar;
		typedef MatrixBase<Matrix<Scalar, Rows, Cols>> base;
		static constexpr int SizeAtCompileTime = Rows * Cols;
		DenseStorage<scalar, SizeAtCompileTime, Rows, Cols> m_data;
	public:
		Matrix(){}
		Matrix(scalar* _array):base(_array){}
		Matrix(std::initializer_list<scalar> _l):base(_l){}
		explicit Matrix(const Matrix& other):m_data(other.m_data){}
		Matrix(Matrix&& other):m_data(std::move(other.m_data)){}

		Matrix& operator=(Matrix& other) {
			return base::operator=(other).derived();
		}

		template<class otherDerived>
		Matrix& operator=(const otherDerived& other) {
			return base::operator=(other).derived();
		}
		scalar& coffeRef(std::size_t idx) {
			return *(m_data.data() + idx);
		}
	};
}
