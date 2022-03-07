#pragma once

#include "common.cuh"
#include "container.cuh"

#include <stdexcept>

namespace bonsai {

  /// SampleDef ///

  struct SampleDef {
    bool numeraire = true;
    Time* forwardMats = NULL;
    Time* discountMats = NULL;
    RateDef* liborDefs = NULL;
    int forwardsSize = 0;
    int discountsSize = 0;
    int liborsSize = 0;

    __host__ ~SampleDef();
  };

  /// Sample<T> ///

  template <class T>
  struct Sample {
    T numeraire;
    T* forwards = NULL;
    T* discounts = NULL;
    T* libors = NULL;
    int forwardsSize = 0;
    int discountsSize = 0;
    int liborsSize = 0;

    __host__ void Allocate(const SampleDef& def);
    __host__ void Initialise();

    __host__ ~Sample();
  };

  /// Scenario<T> ///

  template <class T>
  using Scenario = container<Sample<T>>;

  /// Utility Functions ///

  // Path has already been allocated to have size: defline.
  template <class T>
  inline void AllocatePath(SampleDef const* defline, const int size, 
      Scenario<T>& path);

  template <class T>
  inline void InitialisePath(Scenario<T>& path);

} // namespace bonsai
