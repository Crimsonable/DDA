#include "Matrix.h"
#include "MatrixXpr.h"
#include <vector>
#include <chrono>
#include "Product.h"
#include "eigen/Eigen/Dense"

//#define EIGEN_USE_MKL_ALL

using namespace std;
#define M 5130
#define K 2570
#define N 1290
#define EIGEN_BENCHMARK
using dtype = double;
//#define PRINT_TEST

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
	for (int i = 0; i < M*N; ++i) {
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
	DDA::Matrix<dtype, -1, -1> a, b, c;
	a.resize(N,N); b.resize(N,N); c.resize(N,N);
#ifdef EIGEN_BENCHMARK
	Eigen::Matrix<dtype, -1, -1> ea, eb, ec;
	ea.resize(N, N); eb.resize(N, N); ec.resize(N, N);
	dtype *da = new dtype[N*N], *db = new dtype[N*N], *dc = new dtype[N*N];
#endif // EIGEN_BENCHMARK
	for (int i = 0; i < N*N; ++i) {
		auto f = dtype(i);
		a.coeffRef(i) = f;
		b.coeffRef(i) = f;
		da[i] = f;
		db[i] = f;
#ifdef EIGEN_BENCHMARK
		ea.coeffRef(i) = f;
		eb.coeffRef(i) = f;
#endif // EIGEN_BENCHMARK
	}
	std::cout << "��̬�������" << std::endl;
	std::cout << "-------------------------" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < N*N; ++i)
		dc[i] += da[i] + db[i];
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "��ʱ��" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "���ʽ" << std::endl;
	t0 = std::chrono::steady_clock::now();
	c = a + b;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "��ʱ��" << time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	
#ifdef EIGEN_BENCHMARK
	std::cout << "Eigen" << std::endl;
	t0 = std::chrono::steady_clock::now();
	ec = ea + eb;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "��ʱ��" << time_span.count() << std::endl;
#endif // EIGEN_BENCHMARK
	cout << a.sum() << endl;
	cout << ea.sum() << endl;
}

void test2() {
	double mat[5] = { 1,2,3,4,5 };
	DDA::Matrix<double, 1, 5> a(mat), b(mat), c;
	c = b + a * (b + a);
	c.printMatrix();
}

void test3() {
	DDA::Matrix<dtype, -1, -1> a, b, c;
	Eigen::Matrix<dtype, -1, -1> ea, eb, ec;
	a.resize(M, K); b.resize(K, N); c.resize(M, N);
	ea.resize(M, K); eb.resize(K, N); ec.resize(M, N);
	ea.setRandom(); eb.setRandom();
	c.setZeros();a.setOnes(); b.setOnes();
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < K; ++j) {
			a.coeffRef(i, j) = ea.coeffRef(i, j);
		}
	}
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < N; ++j) {
			b.coeffRef(i, j) = eb.coeffRef(i, j);
		}
	}
	std::cout << "exp" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	t0 = std::chrono::steady_clock::now();
	DDA::Product(&a, &b, &c);
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "cost��" << time_span.count() << std::endl;
	std::cout << "Gflops: " << 1e-9*2 * M*N*K / time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "eigen" << std::endl;
	t0 = std::chrono::steady_clock::now();
	ec = ea * eb;
	t1 = std::chrono::steady_clock::now();
	time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "cost��" << time_span.count() << std::endl;
	std::cout << "Gflops: " << 1e-9*2 * M*N*K / time_span.count() << std::endl;
	cout << ec.sum() << endl;
	cout << c.sum() << endl;
#ifdef PRINT_TEST
	cout << ec << endl;
	c.printMatrix();
#endif // PRINT_TEST
}

void test4() {
	DDA::Matrix<dtype, -1, -1> a, b, c;
	a.resize(M, K); b.resize(K, N); c.resize(M, N);
	c.setZeros(); //a.setOnes(); b.setOnes();
	for (int i = 0; i < M*K; ++i) {
		a.coeffRef(i) = i;
	}
	for (int i = 0; i < K*N; ++i) {
		b.coeffRef(i) = i;
	}
	std::cout << "exp" << std::endl;
	auto t0 = std::chrono::steady_clock::now();
	DDA::Product(&a, &b, &c);
	auto t1 = std::chrono::steady_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
	std::cout << "cost��" << time_span.count() << std::endl;
	std::cout << "Gflops: " << 1e-9 * 2 * M*N*K / time_span.count() << std::endl;
	std::cout << "-------------------------" << std::endl;
	cout << c.sum() << endl;
#ifdef PRINT_TEST
	c.printMatrix();
#endif // PRINT_TEST
}


int main() {
#ifndef _DEBUG
	omp_set_num_threads(4);
#endif
	test3();
	system("pause");
	return 1;
}