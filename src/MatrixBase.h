#pragma once
#include<iostream>
#include <emmintrin.h>
#include "forwardDecleration.h"

namespace DDA {

	template<typename Derived>
	class MatrixBase {
	public:
		using traits = internal::traits<Derived>;

		MatrixBase(){}
		//嵌套类型指：需要模板参数进行进一步推导的类型，前需指定typename，若不指定，编译器将类型当作变量看待
		inline Derived* derived() { return static_cast<Derived*>(this); }

		typename traits::scalar& operator[](std::size_t idx) {
			return derived()->coffeRef(idx);
		}

		MatrixBase& operator=(MatrixBase& other) {
			memcpy(derived()->dataptr(), other.derived()->dataptr(), traits::size * sizeof(typename traits::scalar));
			return *this;
		}

#ifdef DDA_SIMD
		template<typename otherDerived, typename std::enable_if<
											internal::traits<otherDerived>::isXpr, int>::type = 0>
		void operator=(const otherDerived& other) {
			Derived* ptr = derived();
			int vecmod = VECTORIZATION_SIZE / (8 * sizeof(typename traits::scalar)) - ptr->size % (sizeof(typename traits::scalar));
			int real_size = ptr->size + vecmod;
			int endSimd = 8 * real_size * sizeof(typename traits::scalar) / VECTORIZATION_SIZE;
			int strides = VECTORIZATION_SIZE / (8 * sizeof(typename traits::scalar));
			for (int i{}; i < endSimd*strides; i+=strides) {
				__m128 res = other.coffe(i);
				_mm_store_ps(&ptr->coffeRef(i), res);
			}
		}

#else


		template<typename otherDerived, typename std::enable_if<
											internal::traits<otherDerived>::isXpr,int>::type=0>
		void operator=(const otherDerived& other) {
			Derived* ptr = derived();
			for (int i{}; i < ptr->size; ++i)
				ptr->coffeRef(i) = other.coffe(i);
		}
#endif

		void printMatrix() {
			using std::cout, std::endl;
			Derived* ptr = derived();
			for (int i{}; i < ptr->size; i++) {
				cout << ptr->coffeRef(i) << endl;
			}
		}
	};
}