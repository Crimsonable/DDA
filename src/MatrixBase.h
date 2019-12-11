#pragma once
#include "forwardDecleration.h"

namespace DDA {

	template<typename Derived>
	class MatrixBase {
	public:
		using traits = internal::traits<Derived>;

		MatrixBase(){}
		//Ƕ������ָ����Ҫģ��������н�һ���Ƶ������ͣ�ǰ��ָ��typename������ָ���������������͵�����������
		inline Derived* derived() { return static_cast<Derived*>(this); }


		typename traits::scalar& operator[](std::size_t idx) {
			return derived()->coffeRef(idx);
		}

		template<typename otherDerived,typename std::enable_if<
											!internal::traits<otherDerived>::isXpr,int>::type=0>
		void operator=(const otherDerived& other) {
			Derived* ptr = derived();
			ptr->toStorage() = const_cast<otherDerived&>(other).toStorage();
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