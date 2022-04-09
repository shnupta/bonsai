#pragma once

#include <helper_cuda.h>

#include <stdexcept>
#include <assert.h>

namespace bonsai {

  // Basic wrapper around some malloc'd memory
  // Usable both on host and device but must be created on host then copied
  // to device memory
  template <class T>
  class container {
    public:
      __host__ __device__
        container() {};

      // Delete copy contructor and assignment operator (move only)
      __host__ __device__
        container(const container<T>&) = delete;

      __host__ __device__
        container<T>& operator=(const container<T>&) = delete;

      __host__ __device__
        container(container<T>&& c) {
         data_ = c.data_; 
         size_ = c.size_;

         c.data_ = NULL;
         c.size_ = 0;
        }

      __host__ __device__
        container<T>& operator=(container<T>&& c) {
         data_ = c.data_; 
         size_ = c.size_;

         c.data_ = NULL;
         c.size_ = 0;

         return *this;
        }

      __host__ __device__
        container(const int size) : size_(size) {
          data_ = new T[size_];
          assert(data_ != NULL);
        }

      __host__ __device__
        container(const int size, const int internalSize) : size_(size) {
          data_ = new T[size_];
          assert(data_ != NULL);
          for (int i = 0; i < size_; ++i) {
            data_[i] = T(internalSize);
          }
        }

      __host__ __device__// TODO: Double check 
        ~container() {
          /* printf("FUCK"); */
          if (data_ != NULL) delete[] data_;
          size_ = 0;
        }

      __host__ __device__
        void resize(const int size) {
          size_ = size;
          if (data_ != NULL) delete[] data_;
          data_ = new T[size_];
          assert(data_ != NULL);
        }

      __host__ __device__
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
          assert(i < size_ && i >= 0);
          return data_[i];  
        }

      // Setter
      __host__ __device__
        T& operator[](const int i) {
          assert(i < size_ && i >= 0);
          return data_[i];  
        }

      __host__ __device__
        void fill(T value) {
          memset(data_, value, size_ * sizeof(T));
        }

    private:
      int size_ = 0;
      T* data_ = NULL;
  }; // class container

} // namespace bonsai
