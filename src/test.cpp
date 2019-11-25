#include "Matrix.h"
#include "MatrixXpr.h"
#include <vector>
using namespace std;

int main() {
	float mat[5] = { 1,2,3,4,5 };
	DDA::Matrix<float, 1, 5> a(mat);
	DDA::Matrix<float, 1, 5> b(a);
	DDA::Matrix<float, 1, 5> c(mat);
	auto s = a + b*c;
	c = s;
	c.coffeRef(1) = 10;
	//auto s = (a + (b+c));
	c.printMatrix();
	system("pause");
	return 1;
}