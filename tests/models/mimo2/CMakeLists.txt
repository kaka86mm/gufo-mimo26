# Bring-up probes for the mimo2 model: built explicitly, not by routine
# checks, because they need the real artifacts and a GPU.
if (ENGINE_ENABLE_HIP AND GUFO_MIMO2_HIP)
  add_executable(mimo2_dflash_probe EXCLUDE_FROM_ALL dflash_probe.cpp)
  target_link_libraries(mimo2_dflash_probe PRIVATE gufo_mimo2 hip::host)
endif()