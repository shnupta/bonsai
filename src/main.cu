#include "common/sample.cuh"
#include "model/black_scholes.cuh"

using namespace bonsai;

int main() {
  SampleDef def;
  model::BlackScholesModel<double> bsm(100.0, 0.2, false, 0.03, 0.0);

  return 0;
}
