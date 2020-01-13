#include "unitTest.h"
using namespace std;

int main() {
    DDA::Test<double> test;
    test.TestForMatDotPerforemence(9, 9, 5, 1);
    return 1;
}