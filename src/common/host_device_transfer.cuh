#pragma once

#include <type_traits>

/* #include <concepts> */ // Not available on lab compiler yet :'(

namespace bonsai {

  template <class T>
    class RequiresHostDeviceTransfer {
      public:
        __host__ virtual void TransferHostToDevice(T*) const = 0;
        __host__ virtual void TransferDeviceToHost(T*) = 0;
    };


  /* template <class T> */
  /*   struct requires_host_device_transfer { */
  /*     static const bool value = std::is_base_of<RequiresHostDeviceTransfer<T>, T>::value; */
  /*   }; */

  template <class T>
    bool requires_host_device_transfer() {
      return std::is_base_of<RequiresHostDeviceTransfer<T>, T>::value;
    }
}
