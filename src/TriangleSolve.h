#pragma once
#include "forwardDecleration.h"
#include "Matrix.h"
using namespace CSM;

namespace LinerSolver {
	template<typename L_type,typename b_type,typename x_type>
	void DownTriangleSolve(L_type* L, b_type* b, x_type* x) {
		int rows = L->rows, cols = b->cols;
		auto L_data = L->data();
		auto b_data = b->data();
		auto x_data = x->data();
		for (int col_index = 0; col_index < cols; ++col_index) {
			x_data[col_index*rows] = b_data[col_index*rows] / L_data[0];
		}
		Block row_block(L), col_block(x);
		row_block.preAllocMem(1, rows);
		for (int row_index = 1; row_index < rows; ++row_index) {
			row_block.reset(Index(row_index, 0), Index(row_index, row_index - 1));
			row_block.AllocBlockMemcpy();
			row_block.swapshape();
			for (int col_index = 0; col_index < cols; ++col_index) {
				col_block.reset(Index(0, col_index), Index(row_index - 1, col_index));
				double sum = CwiseMulOp(row_block, col_block).sum();
				x_data[row_index + col_index * rows] = (b_data[row_index] - sum) / L_data[row_index + rows * row_index];
			}
		}
	}
}