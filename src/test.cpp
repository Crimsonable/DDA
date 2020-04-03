/*#include <vector>
#include "eigen/Eigen/Dense"
#include "unitTest.h"
#include <fstream>
#include "Rua.h"
#include "Fragment.h"

#define M 2048
#define K 2048
#define N 64
#define NUM_THREADS 4

using namespace std;
using DDA::Index;

void rua() {
	DDA::Matrix<float, -1, -1> a, b, c;
	a.resize(5, 5);
	a.setOnes();
	b.resize(5, 5);
	b.setOnes();
	c.resize(5, 5);
	c.alias() = (a+b)*(a+b);
	c.printMatrix();
}

int main() {
	//Eigen::PartialPivLU<Eigen::Matrix<float, -1, -1>> lu;
	//parameters_choose(128, 512, 8, 50);
	DDA::Test<float> test(M, K, N, NUM_THREADS);
	//auto f = &DDA::Test<float>::TestForMatExpression;
	//auto f = &DDA::Test<float>::TestForMatTranspose;
	auto f = &DDA::Test<float>::TestForMatDotPerforemence;
	test.Loop(50, f);
	//rua();
	system("pause");
    return 1;
}*/