#pragma once

#include "common.cuh"
#include "container.cuh"

#include <stdexcept>

namespace bonsai {

  /// SampleDef ///

  struct SampleDef {
    bool numeraire = true;
    container<Time> forwardMats;
    container<Time> discountMats;
    container<RateDef> liborDefs;

    /* __host__ SampleDef() {} */
    /* __host__ ~SampleDef() {} */
  };

  /// Sample<T> ///

  template <class T>
  struct Sample {
    T numeraire;
    container<T> forwards;
    container<T> discounts;
    container<T> libors;

    __host__ void Allocate(const SampleDef& def) {
      forwards.resize(def.forwardMats.size() * sizeof(T));
      discounts.resize(def.discountMats.size() * sizeof(T));
      libors.resize(def.liborDefs.size() * sizeof(T));
    }

    __host__ void Initialise() {
      numeraire = T(1.0);
      memset(forwards, T(100.0), forwards.size() * sizeof(T));
      memset(discounts, T(1.0), discounts.size() * sizeof(T));
      memset(libors, T(0.0), libors.size() * sizeof(T));
    }
  };

  /// Scenario<T> ///

  template <class T>
  using Scenario = container<Sample<T>>;

  /// Utility Functions ///

  // Path has already been allocated to have size: defline.
  template <class T>
  inline void AllocatePath(SampleDef const* defline, const int size, 
      Scenario<T>& path) {
    for (int i = 0; i < size; ++i) {
      path[i].Allocate(defline[i]);
    }
  }

  template <class T>
  inline void InitialisePath(Scenario<T>& path) {
    for (int i = 0; i < path.Size(); ++i) {
      path[i].Initialise();
    }
  }

} // namespace bonsai
