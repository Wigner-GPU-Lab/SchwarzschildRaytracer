set(PROGRAM_NAME swch_tracer)

file(GLOB SRC *.cpp)

add_executable(${PROGRAM_NAME} ${SRC})
set_property(TARGET ${PROGRAM_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PROGRAM_NAME} sfml-system sfml-window sfml-graphics pthread GL)
target_compile_options(${PROGRAM_NAME} PRIVATE -Wall -Werror)

install(TARGETS ${PROGRAM_NAME}
        DESTINATION bin)
