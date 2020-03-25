#pragma once

namespace DDA {
	struct Index
	{
		int row, col;
		Index(int r, int c):row(r),col(c){}
		Index(){}
	};
}