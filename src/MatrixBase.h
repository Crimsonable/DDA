#pragma once
#include<iostream>
#include<cstring>
#include "forwardDecleration.h"

namespace DDA {

	template<class Derived>
	class MatrixBase {
	public:
		using traits = internal::traits<Derived>;

		MatrixBase(){}
		//Ƕ������ָ����Ҫģ��������н�һ���Ƶ������ͣ�ǰ��ָ��typename������ָ���������������͵�����������
		MatrixBase(typename traits::scalar *_array){
			memcpy(derived().m_data.data(), _array, traits::size*sizeof(*_array));
		}
		MatrixBase(std::initializer_list<typename traits::scalar> _l) {
			for (auto&& i = _l.begin(); i < _l.end(); ++i)
				*(derived().m_data.data() + (i - _l.begin())) = *i;
		}
		inline Derived& derived() { return *static_cast<Derived*>(this); }

		typename traits::scalar& operator[](std::size_t idx) {
			return derived().coffeRef(idx);
		}

		/*MatrixBase& operator=(MatrixBase& other) {
			for (int i{}; i < traits::size; ++i)
				derived().coffeRef(i) = other.derived().coffeRef(i);
			return *this;
		}*/

		template<class otherDerived>
		Derived& operator=(const otherDerived& other) {
			for (int i{}; i < traits::size; ++i)
				derived().coffeRef(i) = std::remove_const_t<otherDerived>(other).coffeRef(i);
			return derived();
		}


		void printMatrix() {
			using std::cout, std::endl;
			for (int i{}; i < traits::size; i++) {
				cout << derived().coffeRef(i) << endl;
			}
		}
	};
}