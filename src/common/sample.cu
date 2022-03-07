#include "sample.cuh"

using namespace bonsai;

/// SampleDef ///

__host__ SampleDef::~SampleDef() {
  if(forwardMats != NULL) free(forwardMats);
  if(discountMats != NULL) free(discountMats);
  if(liborDefs != NULL) free(liborDefs);
}

/// Sample<T> ///

template <class T>
__host__ void Sample<T>::Allocate(const SampleDef& def) {
  forwards = (Time*) malloc(def.forwardsSize * sizeof(T));
  discounts = (Time*) malloc(def.discountsSize * sizeof(T));
  libors = (Time*) malloc(def.liborsSize * sizeof(T));

  if (forwards == NULL || discounts == NULL || libors == NULL) {
    throw std::runtime_error("Sample<T>::Allocate -> malloc failed.");
  }
}

template <class T>
__host__ void Sample<T>::Initialise() {
  numeraire = T(1.0);
  memset(forwards, T(100.0), forwardsSize * sizeof(T));
  memset(discounts, T(1.0), discountsSize * sizeof(T));
  memset(libors, T(0.0), liborsSize * sizeof(T));
}

template <class T>
__host__ Sample<T>::~Sample() {
  if (forwards != NULL) free(forwards);
  if (discounts != NULL) free(discounts);
  if (libors != NULL) free(libors);
}

/// Utility Functions ///

template <class T>
inline void AllocatePath(SampleDef const* defline, int size, 
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
