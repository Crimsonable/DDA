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
			return derived()->coeffRef(idx);
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
            if constexpr(traits::size==-1){
                ptr->resize(other.rows, other.cols);
            }
			const_cast<otherDerived&>(other).toXprBase().run(ptr, other);
		}

#else


		template<typename otherDerived, typename std::enable_if<
											internal::traits<otherDerived>::isXpr,int>::type=0>
		void operator=(const otherDerived& other) {
			Derived* ptr = derived();
			for (int i{}; i < ptr->size; ++i)
				ptr->coeffRef(i) = other.coeff(i);
		}
#endif

		void printMatrix() {
			using std::cout, std::endl;
			Derived* ptr = derived();
			for (int i = 0; i < ptr->rows; ++i) {
				for (int j = 0; j < ptr->cols; ++j) {
					std::cout << ptr->coeffRef(i, j) << " ";
				}
				std::cout << std::endl;
			}
		}

		void printMatrix(int r,int c) {
			using std::cout, std::endl;
			Derived* ptr = derived();
			for (int i = 0; i < r; ++i) {
				for (int j = 0; j < c; ++j) {
					std::cout << ptr->coeffRef(i, j) << " ";
				}
				std::cout << std::endl;
			}
		}
	};
}