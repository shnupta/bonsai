#pragma once

#include "host_device_transfer.cuh"

#include <helper_cuda.h>

#include <stdexcept>
#include <assert.h>

namespace bonsai {

  // Basic wrapper around some malloc'd memory
  // Usable both on host and device but must be created on host then copied
  // to device memory
  template <class T>
  class container : RequiresHostDeviceTransfer<container<T>> {
    public:
      __host__ 
        container() {};

      __host__
        container(const container<T>&) = delete;

      __host__
        container<T>& operator=(const container<T>&) = delete;

      __host__
        container(container<T>&& c) {
         data_ = c.data_; 
         size_ = c.size_;

         c.data_ = NULL;
         c.size_ = 0;
        }

      __host__
        container<T>& operator=(container<T>&& c) {
         data_ = c.data_; 
         size_ = c.size_;

         c.data_ = NULL;
         c.size_ = 0;

         return *this;
        }

      __host__ 
        container(const int size) : size_(size) {
          data_ = new T[size_];
          assert(data_ != NULL);
        }

      __host__
        container(const int size, const int internalSize) : size_(size) {
          data_ = new T[size_];
          assert(data_ != NULL);
          for (int i = 0; i < size_; ++i) {
            data_[i] = T(internalSize);
          }
        }

      __host__ 
        ~container() {
          if (data_ != NULL) delete[] data_;
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
          assert(i < size_ && i >= 0);
          return data_[i];  
        }

      // Setter
      __host__ __device__
        T& operator[](const int i) {
          assert(i < size_ && i >= 0);
          return data_[i];  
        }


        /// RequiresHostToDeviceTransfer ///

        __host__
          void TransferHostToDevice(container<T>* dev) const override {
            // Allocate memory on device for the items of this container
            T* dev_data;
            checkCudaErrors(cudaMalloc((void**) &dev_data, 
                  size_ * sizeof(T)));
            // If we have to copy pointers of the internal object as well
            if (requires_host_device_transfer<T>()) {
              for (int i = 0; i < size_; ++i) {
                // Cast so that we can call relevant transfer functions
                // Bit dodgy with the cast but I can't get the constraints
                // on a trait to show a type requires manual transfer at 
                // compile time
                RequiresHostDeviceTransfer<T>* item = 
                  reinterpret_cast<RequiresHostDeviceTransfer<T>*>(&data_[i]);
                item->TransferHostToDevice(&dev_data[i]);
              }
            } else {
              // Copy all items in data
              checkCudaErrors(cudaMemcpy(dev_data, data_, size_ * sizeof(T),
                    cudaMemcpyHostToDevice));
            }
            // Copy the pointer to memory location from the local to the new
            checkCudaErrors(cudaMemcpy(&(dev->data_), &dev_data, 
                  sizeof(T*), cudaMemcpyHostToDevice));
            // Copy value of size
            checkCudaErrors(cudaMemcpy(&dev->size_, &size_, sizeof(int),
                  cudaMemcpyHostToDevice));
          }

        __host__
          void TransferDeviceToHost(container<T>* dev) override {
            int old_size = size_;
            // Copy size
            checkCudaErrors(cudaMemcpy(&size_, &dev->size_, sizeof(int), 
                  cudaMemcpyDeviceToHost));
            assert(size_ == old_size);
            T* dev_data;
            // Copy pointer
            checkCudaErrors(cudaMemcpy(&dev_data, &(dev->data_), sizeof(T*),
                  cudaMemcpyDeviceToHost));
            if (requires_host_device_transfer<T>()) {
              for (int i = 0; i < size_; ++i) {
                RequiresHostDeviceTransfer<T>* item = 
                  reinterpret_cast<RequiresHostDeviceTransfer<T>*>(
                      &(data_[i]));
                item->TransferDeviceToHost(&dev_data[i]);
              }
            } else {
              // Just copy data at the pointer
              checkCudaErrors(cudaMemcpy(data_, dev_data, size_ * sizeof(T),
                    cudaMemcpyDeviceToHost));
            }
          }

    private:
      int size_ = 0;
      T* data_ = NULL;
  }; // class container

} // namespace bonsai
