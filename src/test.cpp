
#include <vector>
#include "eigen/Eigen/Dense"
#include "unitTest.h"
#include <fstream>
#include "Rua.h"

#define M 2048
#define K 2048
#define N 2048
#define NUM_THREADS 4

using namespace std;
using DDA::Index;

void parameters_choose(int start, int end, int step, int n) {
	using data = std::tuple<double, double, double, double>;
	std::vector<data> dict;
	for (int i = start; i < end; i += step) {
		cout << "total: " << double((i - start) / (end - start)) * 100.0 << '%' << endl;
		DDA::Test<float> test(M, K, N, NUM_THREADS);
		//auto f = &DDA::Test<float>::TestForMatExpression;
		auto f = &DDA::Test<float>::TestForMatDotPerforemence_para;
		dict.push_back(test.Loop(n, f, i));
		system("cls");
	}
	fstream file;
	file.open("para_data.txt", std::ios::in | std::ios::out | std::ios::trunc);
	for (int i = start; i < end; i += step)
		file << i << ' ';
	file << endl;
	for (auto i : dict)
		file << std::get<0>(i) << ' ';
	file << endl;
	for (auto i : dict)
		file << std::get<1>(i) << ' ';
	file << endl;
	for (auto i : dict)
		file << std::get<2>(i) << ' ';
	file << endl;
	for (auto i : dict)
		file << std::get<3>(i) << ' ';
	file.close();
}

int main() {
	//Eigen::PartialPivLU<Eigen::Matrix<float, -1, -1>> lu;
	//parameters_choose(128, 512, 8, 50);
	DDA::Test<float> test(M, K, N, NUM_THREADS);
	//auto f = &DDA::Test<float>::TestForMatExpression;
	//auto f = &DDA::Test<float>::TestForMatTranspose;
	auto f = &DDA::Test<float>::TestForMatDotPerforemence;
	test.Loop(50, f);
	//testForBlock();
	system("pause");
    return 1;
}