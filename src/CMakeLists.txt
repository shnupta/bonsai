find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -G -g") # Debug

add_library(libbonsai
	common/common.cuh
	common/sample.cuh common/sample.cu
	common/container.cuh common/container.cuh
	product/product.cuh
	product/european.cuh
	) 
set_target_properties(libbonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bonsai main.cu)
set_target_properties(bonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bonsai libbonsai -lcurand)
