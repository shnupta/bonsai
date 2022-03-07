#pragma once

#include <stdexcept>

namespace bonsai {

  // Basic wrapper around some malloc'd memory
  // Usable both on host and device but must be created on host then copied
  // to device memory
  template <class T>
  class container {
    public:
      __host__ container(const int size);
      __host__ ~container();

      __host__ __device__ int size() const;

      // Bracket getters and setters
      __host__ __device__ T operator[](const int i) const;
      __host__ __device__ const T& operator[](const int i);

    private:
      int size_ = 0;
      T* data_ = NULL;
  }; // class container

} // namespace bonsai
