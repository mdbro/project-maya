# Writes the production helper (src/mm3/mm3_native_helper.cu) for the CPU helper tests, which compile it as C++ against
# the fake CUDA API in tests/mm3_r2_fake_cuda. The only change: a kernel launch, `kernel<<<config>>>(args)`, which no
# C++ compiler parses, becomes fake_cuda_launch(kernel, config)(args), which the fake API counts. Every other byte is the
# production text. Fails if a launch is left that this rewrite does not recognize.
#   cmake -DIN=<mm3_native_helper.cu> -DOUT=<generated file> -P mm3_helper_fake_source.cmake
file(READ "${IN}" text)
string(REGEX MATCHALL "<<<" before "${text}")
string(REGEX REPLACE "([A-Za-z_][A-Za-z0-9_:]*)<<<([^<>;]*)>>>\\(" "fake_cuda_launch(\\1, \\2)(" text "${text}")
string(FIND "${text}" "<<<" left)
if(NOT left EQUAL -1)
  message(FATAL_ERROR "mm3_helper_fake_source: a kernel launch in ${IN} was not rewritten")
endif()
list(LENGTH before launches)
file(WRITE "${OUT}.tmp" "// Generated from ${IN} by tests/mm3_helper_fake_source.cmake (${launches} kernel launches rewritten).\n#line 1 \"${IN}\"\n${text}")
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
