#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"

namespace DDA {

	namespace internal {
		template<typename Scalar, int Rows, int Cols>
		struct traits<Matrix<Scalar, Rows, Cols>> {
			static constexpr int size = Rows * Cols;
			using scalar = Scalar;
			static constexpr bool isXpr = false;
		}; 

		template<typename Scalar>
		struct traits<Matrix<Scalar, -1, -1>> {
			static constexpr int size = -1;
			using scalar = Scalar;
			static constexpr bool isXpr = false;
		};
	}
	
	template<typename Scalar, int Rows, int Cols>
	class Matrix : public MatrixBase<Matrix<Scalar, Rows, Cols>>, public DenseStorage<Scalar, internal::traits<Matrix<Scalar, Rows, Cols>>::size , Rows, Cols>{
	public:
		using scalar = Scalar;
		typedef MatrixBase<Matrix<Scalar, Rows, Cols>> Matbase;
		typedef DenseStorage<Scalar, internal::traits<Matrix<Scalar, Rows, Cols>>::size, Rows, Cols> Storagebase;
		using traits = internal::traits<Matrix<Scalar, Rows, Cols>>;
	public:
		Matrix(){}
		Matrix(scalar* _array):Storagebase(_array){}
		Matrix(scalar* _array,int _size) :Storagebase(_array,_size) {}
		Matrix(std::initializer_list<scalar> _l):Storagebase(_l){}
		explicit Matrix(const Matrix& other):Storagebase(other.dataptr()){}
		Matrix(Matrix&& other):Storagebase(other.dataptr()){}

		Storagebase& toStorage() {
			return *static_cast<Storagebase*>(this);
		}

		void operator=(const Matrix& other) {
			static_cast<Matbase*>(this)->operator=(other);
		}

		template<typename otherDerived>
		void operator=(const otherDerived& other) {
			static_cast<Matbase*>(this)->operator=(other);
		}

		const scalar& coeff(std::size_t idx) const {
			return *(this->cdata() + idx);
		}

		scalar& coeffRef(std::size_t idx) {
			return *(this->data() + idx);
		}

		scalar& coeffRef(std::size_t r, std::size_t c) {
			return coeffRef(r + c * this->rows);
		}

		void setOnes() {
			auto zeros = new scalar[this->rows];
			for (int j = 0; j < this->rows; ++j)
				zeros[j] = 1;
			auto dataptr = this->data();
			for (int i = 0; i < this->cols; ++i) {
				memcpy(dataptr + i * this->rows, zeros, sizeof(scalar)*this->rows);
			}
			delete[] zeros;
		}

		void setZeros() {
			auto zeros = new scalar[this->rows]{0};
			auto dataptr = this->data();
			for (int i = 0; i < this->cols; ++i)
				memcpy(dataptr + i * this->rows, zeros, sizeof(scalar)*this->rows);
			delete[] zeros;
		}

        void setRandom(){
            auto dataptr = this->data();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<scalar> dis(0, 1);
            for (int i = 0; i < this->size;++i)
                *(dataptr + i) = dis(gen);
        }

		double sum() const {
			double res = 0;
			for (int i = 0; i < this->size; ++i)
				res += this->coeff(i);
			return res;
		}
	};
}
