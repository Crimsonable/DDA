#pragma once
#include<assert.h>
#include<immintrin.h>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<random>
#include<functional>
#include<cstdlib> 
#include<memory>

namespace DDA {
	template<typename Derived> class MatrixBase;
	template<typename T, int size, int Rows, int Cols> class DenseStroage;
	template<typename Scalar, int Rows, int Cols> class Matrix;
	template<typename op, typename lhs, typename rhs> class MatrixXpr;
	template<typename Scalar, typename lhs, typename rhs> class CwiseOpsum;
	template<typename Scalar, typename lhs, typename rhs> class CwiseOpproduct;
	template<typename Scalar, typename other> class CwiseOpscalar;
	template<typename Scalar, typename lhs, typename rhs> class MatDotOp;
	template<typename Self> class TransOp;
	//int Dynamic = -1;
//#define CPUID__AVX2__
#define SIMD
#define CPUID__FAM__
#define VECTORIZATION_ALIGN_BYTES 32
//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_BENCHMARK
//#define DEBUG_INFO
//#define Dynamic -1
#define VEC_CALL  
#define FORCE_INLINE inline
#define PAD_MOD 8

	namespace internal {
		template<typename T> struct traits;
	}
}
