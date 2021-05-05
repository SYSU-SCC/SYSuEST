if (DISTRIBUTED)
    set(QuEST_SRC_ARCHITECTURE_DEPENDENT
        ${CMAKE_CURRENT_SOURCE_DIR}/QuEST_gpu_distributed.cu
        PARENT_SCOPE
    )
else()
    set(QuEST_SRC_ARCHITECTURE_DEPENDENT
        ${CMAKE_CURRENT_SOURCE_DIR}/QuEST_gpu.cu
        PARENT_SCOPE
    )
endif()
