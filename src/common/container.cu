#include "container.cuh"

using namespace bonsai;

template <class T>
__host__
container<T>::container(const int size)  : size_(size) {
  data_ = malloc(size_ * sizeof(T));
  if (data_ == NULL)
      throw std::runtime_error("container<T> -> malloc failed.");
}

template <class T>
__host__
container<T>::~container() {
  if (data_ != NULL) free(data_);
}

template <class T>
__host__ __device__
int container<T>::size() const {
  return size_;
}

template <class T>
__host__ __device__
T container<T>::operator[](const int i) const {
  if (i >= size_)
    throw std::runtime_error("container<T> -> index out of range.");
  return data_[i];
}

template <class T>
__host__ __device__
const T& container<T>::operator[](const int i) {
  if (i >= size_)
    throw std::runtime_error("container<T> -> index out of range.");
  return data_[i];
}
