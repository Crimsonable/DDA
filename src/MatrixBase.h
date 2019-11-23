#pragma once
#include<iostream>
#include "forwardDecleration.h"

namespace DDA {
	namespace internal {
		template<class T> struct traits;
		template<class Scalar, int Rows, int Cols>
		struct traits<Matrix<Scalar, Rows, Cols>> {
			/*enum {
				size = (Rows == Dynamic || Cols == Dynamic) ? 0 : Rows * Cols,
				MatrixKind = (Rows == Dynamic || Cols == Dynamic)
			};*/
			static constexpr int size = Rows * Cols;
			using scalar = Scalar;
		};
	}

	template<class Derived>
	class MatrixBase {
	public:
		using traits = internal::traits<Derived>;

		MatrixBase(){}
		//嵌套类型指：需要模板参数进行进一步推导的类型，前需指定typename，若不指定，编译器将类型当作变量看待
		MatrixBase(typename traits::scalar *_array){
			for (int i{}; i < traits::size; ++i)
				*(derived().m_data.data() + i) = _array[i];
		}
		MatrixBase(std::initializer_list<typename traits::scalar> _l) {
			for (auto&& i = _l.begin(); i < _l.end(); ++i)
				*(derived().m_data.data() + (i - _l.begin())) = *i;
		}
		Derived& derived() { return *static_cast<Derived*>(this); }

		void printMatrix() {
			using std::cout, std::endl;
			auto ptr = derived().m_data.data();
			for (int i{}; i < internal::traits<Derived>::size; i++) {
				cout << ptr[i] << endl;
			}
		}
	};
}