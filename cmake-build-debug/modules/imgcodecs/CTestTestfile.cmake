# CMake generated Testfile for 
# Source directory: /home/truth/github/opencv/modules/imgcodecs
# Build directory: /home/truth/github/opencv/cmake-build-debug/modules/imgcodecs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_imgcodecs "/home/truth/github/opencv/cmake-build-debug/bin/opencv_test_imgcodecs" "--gtest_output=xml:opencv_test_imgcodecs.xml")
set_tests_properties(opencv_test_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Accuracy" WORKING_DIRECTORY "/home/truth/github/opencv/cmake-build-debug/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/truth/github/opencv/cmake/OpenCVUtils.cmake;1649;add_test;/home/truth/github/opencv/cmake/OpenCVModule.cmake;1287;ocv_add_test_from_target;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;148;ocv_add_accuracy_tests;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_perf_imgcodecs "/home/truth/github/opencv/cmake-build-debug/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml")
set_tests_properties(opencv_perf_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Performance" WORKING_DIRECTORY "/home/truth/github/opencv/cmake-build-debug/test-reports/performance" _BACKTRACE_TRIPLES "/home/truth/github/opencv/cmake/OpenCVUtils.cmake;1649;add_test;/home/truth/github/opencv/cmake/OpenCVModule.cmake;1198;ocv_add_test_from_target;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;152;ocv_add_perf_tests;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;0;")
add_test(opencv_sanity_imgcodecs "/home/truth/github/opencv/cmake-build-debug/bin/opencv_perf_imgcodecs" "--gtest_output=xml:opencv_perf_imgcodecs.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Sanity" WORKING_DIRECTORY "/home/truth/github/opencv/cmake-build-debug/test-reports/sanity" _BACKTRACE_TRIPLES "/home/truth/github/opencv/cmake/OpenCVUtils.cmake;1649;add_test;/home/truth/github/opencv/cmake/OpenCVModule.cmake;1199;ocv_add_test_from_target;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;152;ocv_add_perf_tests;/home/truth/github/opencv/modules/imgcodecs/CMakeLists.txt;0;")
