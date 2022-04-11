find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -G -g") # Debug

set(BONSAIHEADERS
	common/common.cuh
	common/sample.cuh
	common/container.cuh
	product/product.cuh
	product/european.cuh
	product/binary_asian.cuh
	model/model.cuh
	model/black_scholes.cuh
	) 
# set_target_properties(libbonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bonsai main.cu ${BONSAIHEADERS})
set_target_properties(bonsai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bonsai -lcurand)
