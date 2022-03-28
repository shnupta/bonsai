#include "common/sample.cuh"
#include "common/container.cuh"
#include "common/host_device_transfer.cuh"
#include "model/black_scholes.cuh"
#include "product/european.cuh"

#include <curand.h>
#include <helper_cuda.h>

#include <stdio.h>

#define THREADBLOCK_SIZE 800

using namespace bonsai;

// Test code

__global__ void TestKernel(container<container<double>>* c) {
  printf("c[0][0] = %f\n", (*c)[0][0]);
  printf("c[1][0] = %f\n", (*c)[1][0]);
}

int main() {
  const double spot = 100.0;
  const double strike = 100.0;
  const double vol = 0.2;
  const double rate = 0.03;
  const double ttm = 1.0 / 12.0; // 1 month
  const bool isSpotMeasure = false; // risk neutral measure
  const int numPaths = 32000;

  product::EuropeanCall<double> call(strike, ttm);
  const int numPayoffs = call.GetPayoffLabels().size();
  container<container<double> > results(numPaths, numPayoffs);

  // No dividend yield
  model::BlackScholesModel<double> bsm(spot, vol, isSpotMeasure, rate); 
  bsm.Allocate(call.GetTimeline(), call.GetDefline());
  bsm.Initialise(call.GetTimeline(), call.GetDefline());

  const int simDim = bsm.GetSimulationDimension();
  printf("simDim = %i\n", simDim);

  /// Random variable generation ///
  const int totalThreads = numPaths * THREADBLOCK_SIZE;
  const int randomVarsCount = numPaths * simDim;
  // Need to allocate numPaths * simDim random variables
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
  curandSetQuasiRandomGeneratorDimensions(gen, simDim);
  double* devZs;
  checkCudaErrors(cudaMalloc((void**) &devZs,
        randomVarsCount * sizeof(double)));
  curandGenerateNormalDouble(gen, devZs, randomVarsCount, 0, 1);
  /// ///

  cudaFree(devZs);
  curandDestroyGenerator(gen);

  container<container<double> > c1(2,1);
  c1[0][0] = 1;
  c1[1][0] = 2;

  container<container<double> >* c2;
  cudaMalloc((void**) &c2, sizeof(container<container<double> >));
  c1.TransferHostToDevice(c2);

  TestKernel<<<1,1>>>(c2);
  cudaDeviceSynchronize();

  return 0;
}
