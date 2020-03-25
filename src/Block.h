#pragma once
#include "forwardDecleration.h"
#include "MatrixBase.h"
#include "DenseStorage.h"

namespace DDA {
	namespace internal {
		template<typename MatType>
		struct traits<Block<MatType>> {
			static constexpr int size = -1;
			using scalar = typename traits<MatType>::scalar;
			static constexpr bool isXpr = false;
			static constexpr bool isDot = false;
		};
	}


	template<typename MatType>
	class Block :public MatrixBase<Block<MatType>>, 
				 public DenseStorage<typename internal::traits<MatType>::scalar, -1, -1, -1> {
	private:
		using scalar = typename internal::traits<MatType>::scalar;
		typedef MatrixBase<Block<scalar>> Matbase;
		typedef DenseStorage<scalar, -1, -1, -1> Storagebase;
		Index topLeft, bottomRight;
		int offset, master_rows, master_cols, padRows = 0;
		bool isAllocate = false;
		MatType* mat;
	public:
		Block(MatType& mat, Index&& topLeft, Index&& bottomRight):mat(&mat) {
			reset(std::forward<Index>(topLeft), std::forward<Index>(bottomRight));
			this->share(&mat);
		}
		Block(MatType& mat):mat(&mat){}

		inline Storagebase& toStorage(){
			return *static_cast<Storagebase*>(const_cast<Block*>(this));
		}

		inline scalar* data() {
			return toStorage().data() + offset;
		}

		inline const scalar& coeff(std::size_t idx) const {
			return *(this->cdata() + idx / rows * master_rows + idx % rows + offset);
		}

		inline scalar& coeffRef(std::size_t idx) {
			return *(data() + idx / rows * master_rows + idx % rows);
		}

		inline scalar& coeffRef(std::size_t r, std::size_t c) {
			return coeffRef(r + c * this->rows);
		}

		void AllocBlockMemcpy() {
			auto master_ptr = data();
			offset = 0;
			this->m_storage = std::make_shared<typename internal::plain_array<scalar, internal::traits<MatType>::size>>(size);
			this->resize(rows, cols);
			for (int offset_col = 0; offset_col < cols; ++offset_col) {
				auto dst = data() + offset_col * rows;
				auto src = master_ptr + offset_col * master_cols;
				memcpy(dst, src, sizeof(scalar)*(rows - padRows));
			}
			master_rows = rows;
			master_cols = cols;
			isAllocate = true;
		}

		inline void reset(Index&& topLeft, Index&& bottomRight) {
			this->topLeft = topLeft;
			this->bottomRight = bottomRight;
			rows = bottomRight.row - topLeft.row + 1;
			cols = bottomRight.col - topLeft.col + 1;
			size = rows * cols;
			master_rows = mat->rows;
			master_cols = mat->cols;
			offset = topLeft.row + topLeft.col*master_rows;
			padRows = bottomRight.row - master_rows + 1;
		}

		int rows, cols;
		std::size_t size;
	};
}