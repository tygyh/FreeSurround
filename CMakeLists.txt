cmake_minimum_required(VERSION 3.29)
project(FreeSurround)

set(CMAKE_CXX_STANDARD 23)

add_library(FreeSurround STATIC
        source/ChannelMaps.cpp
        include/FreeSurround/_KissFFTGuts.h
        include/FreeSurround/ChannelMaps.h
        source/KissFFT.cpp
        source/FreeSurroundDecoder.cpp
        source/KissFFTR.cpp)
