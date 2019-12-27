#include "Matrix.h"
#include "MatrixXpr.h"
#include <vector>
#include <chrono>
#include "Product.h"
#include "eigen/Eigen/Dense"
using namespace std;
#define N 2048
#define N2 2048
//#define EIGEN_BENCHMARK

template<typename T>
bool check1(T& mat) {
	for (int i = 0; i < mat.size; ++i) {
		if (mat.coeff(i) != N)
			return false;
	}
	return true;
}

template<typename T1, typename T2>
void check(T1& ec, T2& c) {
	for (int i = 0; i < N2*N2; ++i) {
		if (abs(ec.coeffRef(i) - c.coeffRef(i)) > 1e-6) {
			//cout << "false at " << i % N2 << "  " << i / N2 << endl;
			cout << 0 << endl;
			return;
		}
	}
	cout << 1 << endl;
}

template<typename T>
void check0(T& c) {
	for (int i = 0; i < c.size; ++i) {
		if (abs(c.coeffRef(i) - 0) > 1e-6)
			cout << "false initialize" << std::endl;
	}
}

void test() {
	DDA::Matrix<float, -1, -1> a, b, c;
	a.resize(N,N); b.resize(N,N); c.resize(N,N);
	//DDA::Matrix<float, N, N> a, b, c;
#ifdef EIGEN_BENCHMARK
	//Eigen::Matrix<float, N, N> ea, eb, ec;
	Eigen::Matrix<float, -1, -1> ea, eb, ec;
	ea.resize(N, N); eb.resize(N, N); ec.resize(N, N);
#endif // EIGEN_BENCHMARK
	float sa[N*N], sb[N*N], sr[N*N];
	float *da = new float[N*N];
	float *db = new float[N*N];
	float *dc = new float[N*N];
	for (int i = 0; i < N*N; ++i) {
		auto f = float(i);
		a.coeffRef(i) = f;
		b.coeffRef(i) = f;
		sa[i] = f;
		sb[i] = f;
		da[i] = f;
		db[i] = f;
#ifdef EIGEN_BENCHMARK
		ea.coeffRef(i) = f;
		eb.coeffRef(i) = f;
#endif // EIGEN_BENCHMARK
	}
	std::cout << "堆数组直接相加" << std::endl;
	std::cout << "-------------------------" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N*N; ++i)
		sr[i] += sa[i] + sb[i];
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "动态数组" << std::endl;
	t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N*N; ++i)
		dc[i] += db[i] + da[i];
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "表达式" << std::endl;
	t0 = std::chrono::steady_clock::now();
	c = a + b;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	
#ifdef EIGEN_BENCHMARK
	std::cout << "Eigen" << std::endl;
	t0 = std::chrono::steady_clock::now();
	ec = ea + eb;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "用时：" << time_span.count() << std::endl;
#endif // EIGEN_BENCHMARK
}

void test2() {
	float mat[5] = { 1,2,3,4,5 };
	DDA::Matrix<float, -1, -1> a;
	DDA::Matrix<float, 1, 5> b(mat), c(mat);
	a = b + c * (b + c);
	a.printMatrix();
}

void test3() {
	DDA::Matrix<float, -1, -1> a, b, c;
	Eigen::Matrix<float, -1, -1> ea, eb, ec;
	a.resize(N, N2); b.resize(N2, N); c.resize(N, N);
	ea.resize(N, N2); eb.resize(N2, N); ec.resize(N, N);
	ea.setOnes(); eb.setOnes(); ec.setZero();
	c.setZeros();a.setOnes(); b.setOnes();
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N2; ++j) {
			a.coeffRef(i, j) = j + i;
			ea.coeffRef(i, j) = j + i;
		}
	}
	for (int i = 0; i < N2; ++i) {
		for (int j = 0; j < N; ++j) {
			b.coeffRef(i, j) = j + i;
			eb.coeffRef(i, j) = j + i;
		}
	}
	std::cout << "exp" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	DDA::Product(&a, &b, &c);
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "cost：" << time_span.count() << std::endl;
	std::cout << "Gflops: " << 1e-9*2 * N*N*N / time_span.count() << std::endl;
	//std::cout << check1(c) << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "eigen" << std::endl;
	t0 = std::chrono::steady_clock::now();
	ec = ea * eb;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "cost：" << time_span.count() << std::endl;
	std::cout << "Gflops: " << 1e-9*2 * N*N*N / time_span.count() << std::endl;
	//check(ec, c);
	cout << ec.sum() << endl;
	cout << c.sum() << endl;
	//cout << ec << endl;
	//c.printMatrix();
}

void test4() {
	DDA::Matrix<float, -1, -1> a, b, c;
	Eigen::Matrix<float, -1, -1> ea, eb, ec;
	ea.resize(N, N2); eb.resize(N2, N2); ec.resize(N, N2); ec.setZero();
	a.resize(N, N2); b.resize(N2, N2); c.resize(N, N2);c.setZeros();
	/*a.setOnes(); b.setOnes();
	ea.setOnes(); eb.setOnes();*/
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N2; ++j) {
			a.coeffRef(i, j) = j;
			ea.coeffRef(i, j) = j;
		}
	}
	for (int i = 0; i < N2; ++i) {
		for (int j = 0; j < N; ++j) {
			b.coeffRef(i, j) = j;
			eb.coeffRef(i, j) = j;
		}
	}
	DDA::Product(&a, &b, &c);
	ec = ea * eb;
	auto correct = abs(ec.sum() - c.sum());
	cout << ec.sum() << endl;
	cout << c.sum() << endl;
	check(ec, c);
	//cout << ec << endl;
	//c.printMatrix();
}


int main() {
#ifndef _DEBUG
	omp_set_num_threads(6);
#endif
	test3();
	system("pause");
	return 1;
}