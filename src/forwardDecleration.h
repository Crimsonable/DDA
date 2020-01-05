#pragma once
#include<assert.h>
#include <immintrin.h>
#include<iostream>
namespace DDA {
	template<typename Derived> class MatrixBase;
	template<typename T, int size, int Rows, int Cols> class DenseStroage;
	template<typename Scalar, int Rows, int Cols> class Matrix;
	template<typename op, typename lhs, typename rhs> class MatrixXpr;
	template<typename Scalar, typename lhs, typename rhs> class CwiseOpsum;
	template<typename Scalar, typename lhs, typename rhs> class CwiseOpproduct;
	template<typename Scalar, typename other> class CwiseOpscalar;
	template<typename Scalar, typename lhs, typename rhs> class ProductOp;
	//int Dynamic = -1;
#define DDA_SIMD
#define VECTORIZATION_SIZE 128
//#define Dynamic -1

	namespace internal {
		template<typename T> struct traits;
	}
}
