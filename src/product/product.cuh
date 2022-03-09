#pragma once

#include "../common/common.cuh"
#include "../common/container.cuh"
#include "../common/sample.cuh"

namespace bonsai {
  namespace product {

    template <class T>
    class Product {
      public:
        __host__ __device__
          virtual const container<Time>& GetTimeline() const = 0; 
        __host__ __device__
          virtual const container<SampleDef>& GetDefline() const = 0;

        __host__
          virtual const container<std::string>& GetPayoffLabels() const = 0;

        // TODO: Define some __device__ only ComputePayoffs function that 
        // works stuff out thread per thread and accumulates the value

        // TODO: What's the point of the copy constructor?
    };

  } // namespace product
} // namespace bonsai
