
set(CF_PIOTR_SOURCES
    ${CF_PIOTR_DIR}/gradientMex.hpp
    ${CF_PIOTR_DIR}/src/gradientMex.cpp
    ${CF_PIOTR_DIR}/src/sse.hpp
    ${CF_PIOTR_DIR}/src/wrappers.hpp)

set(CF_LIB_COMMON_SOURCES
     cf_libs/common/feature_channels.hpp
     cf_libs/common/mat_consts.hpp
     cf_libs/common/math_helper.hpp
     cf_libs/common/math_helper.cpp
     cf_libs/common/cf_tracker.hpp
     cf_libs/common/tracker_debug.hpp
     cf_libs/common/scale_estimator.hpp
     cf_libs/common/cv_ext.hpp
     cf_libs/kcf/kcf_debug.hpp
     cf_libs/kcf/kcf_tracker.hpp
     cf_libs/dsst/dsst_debug.hpp
     cf_libs/dsst/dsst_tracker.hpp
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/shift.hpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.hpp
    ${CF_CV_EXT_DIR}/psr.hpp
    ${CF_PIOTR_SOURCES})

add_library(kcf
    ${CF_LIB_COMMON_SOURCES}
    ${CF_PIOTR_SOURCES})
target_link_libraries(kcf ${OpenCV_LIBS})


