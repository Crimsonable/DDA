#include "Matrix.h"
#include "MatrixXpr.h"
#include <vector>
#include <chrono>
//#include "eigen/Eigen/Dense"
using namespace std;
#define N 10000


void test() {
	DDA::Matrix<float, 1, N> a, b, c;
	//Eigen::Matrix<float, 1, N> ea, eb, ec;
	float sa[N], sb[N], sr[N];
	for (int i = 0; i < N; ++i) {
		a.coffeRef(i) = float(i);
		b.coffeRef(i) = float(i);
		//ea.coeffRef(0, i) = float(i);
		//eb.coeffRef(0, i) = float(i);
		sa[i] = i;
		sb[i] = i;
	}

	std::cout << "normal array" << std::endl;
	std::cout << "-------------------------" << std::endl;
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i)
		sr[i] = sa[i] + sb[i];
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span = t1 - t0;
	std::cout << "time cost: " << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "expression" << std::endl;
	t0 = std::chrono::high_resolution_clock::now();
	c = a + b;
	t1 = std::chrono::high_resolution_clock::now();
	time_span = t1 - t0;
	std::cout << "time cost: " << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	/*std::cout << "Eigen: " << std::endl;
	t0 = std::chrono::steady_clock::now();
	ec = ea + eb;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "��ʱ��" << time_span.count() << std::endl;*/
}

void test2() {
	float mat[5] = { 1,2,3,4,5 };
	DDA::Matrix<float,1, 5> a(mat), b(mat), c;
	c = a + b;
	c.printMatrix();
}

int main() {
	test();
	system("pause");
	return 1;
}