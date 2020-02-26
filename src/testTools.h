#pragma once
#include<iostream>
#include<string>
using namespace std;

namespace DEBUG_TOOLS {
	template<typename T>
	void printRawMatrix(T *data, int rows, int cols, string info="DEBUG INFO") {
		std::cout << info << std::endl;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j)
				cout << data[i + j * rows] << " ";
			cout << endl;
		}
		cout << endl;
	}
}