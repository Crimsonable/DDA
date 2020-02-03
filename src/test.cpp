#include <chrono>
#include <vector>
#include "Matrix.h"
#include "MatrixXpr.h"
#include "Product.h"
#include "eigen/Eigen/Dense"
#include "unitTest.h"

using namespace std;

int main(int argc, char **argv) {
    DDA::Test<float> test;
    test.Loop(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
    return 1;
}