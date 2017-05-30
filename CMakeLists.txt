project(Research)

cmake_minimum_required(VERSION 2.8)

option(BUILD_SCHWARZSCHILD_RAYTRACER "Build the Schwarzschild Raytracer" ON)

if (${BUILD_SCHWARZSCHILD_RAYTRACER})
    add_subdirectory(SchwarzschildRaytracer)
endif ()
