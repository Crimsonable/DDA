#pragma once

namespace CSM {
	struct Index
	{
		int row, col;
		Index(int r, int c):row(r),col(c){}
		Index(){}
		bool operator<(const Index& other) {
			return other.row > row && other.col > col;
		}
		bool operator>(const Index& other) {
			return row > other.row && col > other.col;
		}
	};
}