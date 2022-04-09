#pragma once

#include "common.cuh"
#include "container.cuh"

#include <stdexcept>

namespace bonsai {

  /// SampleDef ///

  // Definition of the data that we must simulate
  struct SampleDef {
    // true for payment dates, false otherwise
    bool numeraire = true;
    // Maturities of the forwards on this event date
    container<Time> forwardMats;
    // Maturities of the discounts on this event date
    container<Time> discountMats;
    // Specification of the libors on this event date
    container<RateDef> liborDefs;

    /* __host__ SampleDef() {} */
    /* __host__ ~SampleDef() {} */
  };

  /// Sample<T> ///

  // A sample is the collection of market observations on an event date for the
  // evaluation of the payoff
  template <class T>
  struct Sample {
    T numeraire;
    container<T> forwards;
    container<T> discounts;
    container<T> libors;

    // Allocate given a corresponding definition
    __host__ __device__ void Allocate(const SampleDef& def) {
      forwards.resize(def.forwardMats.size());
      discounts.resize(def.discountMats.size());
      libors.resize(def.liborDefs.size());
    }

    // Initialise defaults
    __host__ __device__ void Initialise() {
      numeraire = T(1.0);
      forwards.fill(T(100.0));
      discounts.fill(T(1.0));
      libors.fill(T(0.0));
    }
  };

  /// Scenario<T> ///

  // Scenarios are collections of samples
  // They are the objects that models and products use to communicate
  template <class T>
  using Scenario = container<Sample<T>>;

  /// Utility Functions ///

  // Batch allocate a collection of samples
  // Path has already been allocated to have size: defline.
  template <class T>
  __host__ __device__
  /*inline*/ void AllocatePath(const container<SampleDef>& defline,
      Scenario<T>& path) {
    path.resize(defline.size());
    for (int i = 0; i < defline.size(); ++i) {
      path[i].Allocate(defline[i]);
    }
  }

  // Batch initialise a collection of samples
  template <class T>
  __host__ __device__
  /*inline*/ void InitialisePath(Scenario<T>& path) {
    for (int i = 0; i < path.size(); ++i) {
      path[i].Initialise();
    }
  }

} // namespace bonsai
