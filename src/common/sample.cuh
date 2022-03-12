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
    __host__ void Allocate(const SampleDef& def) {
      forwards.resize(def.forwardMats.size() * sizeof(T));
      discounts.resize(def.discountMats.size() * sizeof(T));
      libors.resize(def.liborDefs.size() * sizeof(T));
    }

    // Initialise defaults
    __host__ void Initialise() {
      numeraire = T(1.0);
      memset(forwards, T(100.0), forwards.size() * sizeof(T));
      memset(discounts, T(1.0), discounts.size() * sizeof(T));
      memset(libors, T(0.0), libors.size() * sizeof(T));
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
  inline void AllocatePath(SampleDef const* defline,
      Scenario<T>& path) {
    for (int i = 0; i < path.size(); ++i) {
      path[i].Allocate(defline[i]);
    }
  }

  // Batch initialise a collection of samples
  template <class T>
  inline void InitialisePath(Scenario<T>& path) {
    for (int i = 0; i < path.size(); ++i) {
      path[i].Initialise();
    }
  }

} // namespace bonsai
