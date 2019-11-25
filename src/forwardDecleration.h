#pragma once
namespace DDA {
	template<class Derived> class MatrixBase;
	template<class Scalar, int Rows, int Cols> class Matrix;
	template<class op, class lhs, class rhs> class MatrixXpr;
	template<class Scalar, class lhs, class rhs> class CwiseOpsum;
	template<class Scalar, class lhs, class rhs> class CwiseOpproduct;
	int Dynamic = -1;

	namespace internal {
		template<class T> struct traits;
	}
}
