#include "unitTest.h"
using namespace std;

int main() {
    DDA::Test<double> test;
    test.TestForMatDotPerforemence(517, 259, 135, 4);
    return 1;
}