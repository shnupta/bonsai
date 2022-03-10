#pragma once

#include <stdexcept>
#include <assert.h>

namespace bonsai {

  // Basic wrapper around some malloc'd memory
  // Usable both on host and device but must be created on host then copied
  // to device memory
  template <class T>
  class container {
    public:
      __host__ 
        container() {};

      __host__ 
        container(const int size) : size_(size) {
          data_ = new T[size_];
          assert(data_ != NULL);
        }

      __host__ 
        ~container() {
          delete[] data_;
          size_ = 0;
        }

      __host__ 
        void resize(const int size) {
          size_ = size;
          if (data_ != NULL) delete[] data_;
          data_ = new T[size_];
          assert(data_ != NULL);
        }

      __host__
        void clear() {
          size_ = 0;
          if (data_ != NULL) delete[] data_;
        }

      __host__ __device__
        int size() const {
          return size_;
        }

      // Getter (note the const)
      __host__ __device__
        const T& operator[](const int i) const {
          assert(i < size_ || i >= 0);
          return data_[i];  
        }

      // Setter
      __host__ __device__
        T& operator[](const int i) {
          assert(i < size_ || i >= 0);
          return data_[i];  
        }

    private:
      int size_ = 0;
      T* data_ = NULL;
  }; // class container

} // namespace bonsai
