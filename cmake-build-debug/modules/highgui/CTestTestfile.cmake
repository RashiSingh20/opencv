# CMake generated Testfile for 
# Source directory: /home/truth/github/opencv/modules/highgui
# Build directory: /home/truth/github/opencv/cmake-build-debug/modules/highgui
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_highgui "/home/truth/github/opencv/cmake-build-debug/bin/opencv_test_highgui" "--gtest_output=xml:opencv_test_highgui.xml")
set_tests_properties(opencv_test_highgui PROPERTIES  LABELS "Main;opencv_highgui;Accuracy" WORKING_DIRECTORY "/home/truth/github/opencv/cmake-build-debug/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/truth/github/opencv/cmake/OpenCVUtils.cmake;1649;add_test;/home/truth/github/opencv/cmake/OpenCVModule.cmake;1287;ocv_add_test_from_target;/home/truth/github/opencv/modules/highgui/CMakeLists.txt;168;ocv_add_accuracy_tests;/home/truth/github/opencv/modules/highgui/CMakeLists.txt;0;")
