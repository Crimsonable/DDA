#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"

namespace DDA {
	

	template<class Scalar, int Rows, int Cols>
	class Matrix : public MatrixBase<Matrix<Scalar, Rows, Cols>> {
	public:
		using scalar = Scalar;
		typename MatrixBase<Matrix<Scalar, Rows, Cols>> base;
		static constexpr int SizeAtCompileTime = Rows * Cols;
		DenseStorage<scalar, SizeAtCompileTime, Rows, Cols> m_data;
	public:
		Matrix(){}
		Matrix(scalar* _array):base(_array){}
		Matrix(std::initializer_list<scalar> _l):base(_l){}
		Matrix(const Matrix& other):m_data(other.m_data){}
		Matrix(Matrix&& other):m_data(std::move(other.m_data)){}
	};
}
