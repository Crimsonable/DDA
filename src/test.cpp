#include "Matrix.h"
#include <vector>
using namespace std;

int main() {
	float a[5] = { 1,2,3,4,5 };
	DDA::Matrix<float, 1, 5> mat(a);
	DDA::Matrix<float, 1, 5> b({1,2,3,4,5});
	b.printMatrix();
	mat.printMatrix();
	system("pause");
	return 1;
}