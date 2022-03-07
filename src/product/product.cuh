#pragma once

#include "../common/common.cuh"
#include "../common/container.cuh"

namespace bonsai {
  namespace product {

    template <class T>
    class Product {
      public:
        __host__ __device__
          virtual const container<Time>& GetTimeline() const = 0; 
        __host__ __device__
          virtual const container<Time>& GetDefline() const = 0;

        __host__
          virtual const container<std::string>& GetPayoffLabels() const = 0;

        // TODO: Don't think I need the ComputePayoffs function here
        // That should be some sort of kernel function, so as long as I have 
        // the path and the output container I can calculate them on device

        // TODO: What's the point of the copy constructor?
    };

  } // namespace product
} // namespace bonsai
