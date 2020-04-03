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

#include "Index.h"

//#define CPUID__AVX2__
#define SIMD
#define CPUID__FAM__
#define VECTORIZATION_ALIGN_BYTES 32
#define EIGEN_USE_MKL_ALL
//#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_BENCHMARK
//#define DEBUG_INFO
//#define Dynamic -1
#define VEC_CALL __vectorcall
#define FORCE_INLINE __forceinline
#define PAD_MOD 8
//#define	_ENABLE_EXTENDED_ALIGNED_STORAGE
//#define HAS_CUDA

namespace DDA {
	template<typename Derived> class MatrixBase;
	//template<typename T, int size, int Rows, int Cols> class DenseStroage;
	template<typename Scalar, int Rows, int Cols> class Matrix;
	/*template<typename op, typename lhs, typename rhs> class MatrixXpr;
	template<typename Scalar, typename other> class CwiseOpscalar;*/
	template<typename Self> class TransOp;
	template<typename SolverType> class SolverBase;
	template<typename MatType> class Iterator;
	template<typename Scalar> class Block;

	template<typename Lhs, typename Rhs> class MatMulOp;
	template<typename Lhs, typename Rhs, typename Function> class CwiseBaseOp;
	template<typename Lhs, typename Rhs> class CwiseSumOp;
	template<typename Lhs, typename Rhs> class CwiseSubOp;
	template<typename Lhs, typename Rhs> class CwiseMulOp;

	class CommonBase{};

	namespace internal {
		template<typename T> struct traits;
	}
}
