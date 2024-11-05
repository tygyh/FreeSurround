cmake_minimum_required(VERSION 3.5)
project(FreeSurround)

# Set Boost root directory
set(BOOST_ROOT "../boost_1_86_0")

# Link Boost libraries
target_link_libraries(freesurround_exe Boost::filesystem Boost::thread)

if (NOT MSVC)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

set(SRCS
  source/ChannelMaps.cpp
  source/KissFFT.cpp
  source/KissFFTR.cpp
  source/FreeSurroundDecoder.cpp
)
