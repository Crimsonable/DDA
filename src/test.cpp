#include <vector>
#include "eigen/Eigen/Dense"
#include "unitTest.h"
#include <fstream>
#include "Rua.h"
#include "Fragment.h"
#include "TriangleSolve.h"

#define M 2048
#define K 2048
#define N 2048
#define NUM_THREADS 4

using namespace std;
using CSM::Index;
using namespace CPU_OP;

void rua() {
	Matrix<float, -1, -1> a, b, c;
	a.resize(5, 6);
	a.setRandom();
	b.resize(6, 5);
	b.setRandom();
	auto exp = (a*(b+a.transpose())).toExpression();
	c = exp;
	//a.printMatrix();
	c.printMatrix();
	exp.clear();
}

int main() {
	/*CSM::Test<float> test(M, K, N, NUM_THREADS);
	test.Bench(50);*/
	rua();
	system("pause");
    return 1;
}