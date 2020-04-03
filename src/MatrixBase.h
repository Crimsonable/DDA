#pragma once
#include "forwardDecleration.h"
#include "Iterator.h"

namespace DDA {

	template<typename Derived>
	class MatrixBase :CommonBase
	{
	private:
		using traits = internal::traits<Derived>;
		using scalar = typename traits::scalar;
		bool force_lazy = false;
		Derived *ptr = derived();
		inline Derived* derived() { return static_cast<Derived*>(this); }

	public:
		MatrixBase(){}

		typename traits::scalar& operator[](std::size_t idx) {
			return ptr->coeffRef(idx);
		}

		template<typename otherDerived,typename std::enable_if<
											!internal::traits<otherDerived>::isXpr,int>::type=0>
		void operator=(const otherDerived& other) {
			ptr->toStorage() = const_cast<otherDerived&>(other).toStorage();
		}

		template<typename otherDerived, typename std::enable_if<
											internal::traits<otherDerived>::isXpr, int>::type = 0>
		void operator=(const otherDerived& other) {
            if constexpr(traits::size==-1){
                ptr->resize(other.rows, other.cols);
            }
			const_cast<otherDerived&>(other).toXprBase().run(ptr, other, force_lazy);
			force_lazy = false;
		}

		inline Derived& alias() {
			force_lazy = true;
			return *ptr;
		}

		Iterator<Derived> begin() {
			auto it = Iterator(*ptr);
			it.begin();
			return it;
		}

		Iterator<Derived> end() {
			auto it = Iterator(*ptr);
			it.end();
			return it;
		}

		void printMatrix() {
			for (int i = 0; i < ptr->rows; ++i) {
				for (int j = 0; j < ptr->cols; ++j) {
					std::cout << ptr->coeffRef(i, j) << " ";
				}
				std::cout << std::endl;
			}
		}

		void printMatrix(int r,int c) {
			for (int i = 0; i < r; ++i) {
				for (int j = 0; j < c; ++j) {
					std::cout << ptr->coeffRef(i, j) << " ";
				}
				std::cout << std::endl;
			}
		}

		void setOnes() {
			auto zeros = new scalar[ptr->rows];
			for (int j = 0; j < ptr->rows; ++j)
				zeros[j] = 1;
			auto dataptr = ptr->data();
			for (int i = 0; i < ptr->cols; ++i) {
				memcpy(dataptr + i * ptr->rows, zeros, sizeof(scalar)*ptr->rows);
			}
			delete[] zeros;
		}

		void setZeros() {
			auto zeros = new scalar[ptr->rows]{ 0 };
			auto dataptr = ptr->data();
			for (int i = 0; i < ptr->cols; ++i)
				memcpy(dataptr + i * ptr->rows, zeros, sizeof(scalar)*ptr->rows);
			delete[] zeros;
		}

		void setRandom() {
			auto dataptr = ptr->data();
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<scalar> dis(0, 1);
			for (auto &i:*ptr)
				i = dis(gen);
		}

		double sum() const {
			double res = 0;
			for (auto i : *ptr)
				res += i;
			return res;
		}
	};
}