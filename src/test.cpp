#include "Matrix.h"
#include "MatrixXpr.h"
#include <vector>
#include <chrono>
using namespace std;
#define N 20000


void test() {
	DDA::Matrix<float, -1, -1> a, b, c;
	a.resize(N); b.resize(N); c.resize(N);
	//DDA::Matrix<float, 1, N> a, b, c;
	float sa[N], sb[N], sr[N];
	float *da = new float[N];
	float *db = new float[N];
	float *dc = new float[N];
	for (int i = 0; i < N; ++i) {
		auto f = float(i);
		a.coffeRef(i) = f;
		b.coffeRef(i) = f;
		sa[i] = f;
		sb[i] = f;
		da[i] = f;
		db[i] = f;
	}

	std::cout << "静态数组" << std::endl;
	std::cout << "-------------------------" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N; ++i)
		sr[i] = sa[i] * sb[i];
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "动态数组" << std::endl;
	t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N; ++i)
		dc[i] = db[i] * da[i];
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "表达式ʽ" << std::endl;
	t0 = std::chrono::steady_clock::now();
	c = a * b;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
}

void test2() {
	float mat[5] = { 1,2,3,4,5 };
	DDA::Matrix<float, -1, -1> a;
	DDA::Matrix<float, 1, 5> b(mat);
	a = b;
	a.printMatrix();
}

int main() {
	test2();
	system("pause");
	return 1;
}