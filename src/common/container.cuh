#pragma once

#include <stdexcept>

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
        if (data_ == NULL)
          throw std::runtime_error("container<T>::constructor -> new failed.");
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
        if (data_ == NULL)
          throw std::runtime_error("container<T>::resize -> new failed.");
      }

      __host__ __device__
        int size() const {
          return size_;
        }

      // Bracket getters and setters
      __host__ __device__
      T operator[](const int i) const {
        if (i >= size_)
            throw "container<T>::[] -> index out of range.";
        return data_[i];  
      }

      __host__ __device__
      T& operator[](const int i) {
        if (i >= size_)
            throw "container<T>::[] -> index out of range.";
        return data_[i];  
      }

    private:
      int size_ = 0;
      T* data_ = NULL;
  }; // class container

} // namespace bonsai
