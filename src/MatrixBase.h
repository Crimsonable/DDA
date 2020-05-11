#pragma once
#include "forwardDecleration.h"
#include "Iterator.h"
#include "OpRegister.h"
#include "MetaTools.h"
/*#include <cuda_runtime.h>
#include "CublasProduct.h"*/

namespace CSM {
	enum Device
	{
		CPU,GPU
	};

	template<typename Derived>
	class MatrixBase:public CommonBase
	{
	private:
		using scalar = typename internal::traits<Derived>::scalar;
		Derived *ptr = derived();
		inline Derived* derived() { return static_cast<Derived*>(this); }
	public:
		Device device = CPU;

		template<typename otherDerived,typename std::enable_if<
											!internal::traits<otherDerived>::isXpr,int>::type=0>
		void operator=(const otherDerived& other) {
			ptr->toStorage() = const_cast<otherDerived&>(other).toStorage();
		}

		inline Derived& alias() {
			ptr->defaultLazyAssign = true;
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

		void setEye() {
#ifdef _DEBUG
			if (ptr->rows != ptr->cols) {
				std::cout << "Matrix must be square!" << std::endl;
				abort();
			}
#endif // _DEBUG
			auto dataptr = ptr->data();
			for (int i = 0; i < ptr->rows; ++i)
				dataptr[i + i * ptr->rows] = scalar(1);
		}

		void setTriangleDownRandom() {
			auto dataptr = ptr->data();
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<scalar> dis(0, 1);
			for (int col_index = 0; col_index < ptr->cols; ++col_index) {
				for (int row_index = col_index; row_index < ptr->rows; ++row_index) {
					dataptr[col_index*ptr->rows + row_index] = dis(gen);
				}
			}
		}

		double sum() const {
			double res = 0;
			for (auto i : *ptr)
				res += i;
			return res;
		}

		int toGPU() {
			device = GPU;
			//CUDA_TOOLS::MemCopyToGpu(ptr->data(), sizeof(scalar)*ptr->size);
		}
	};
}