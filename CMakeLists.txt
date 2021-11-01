cmake_minimum_required(VERSION 3.17)

project(qedic)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_AUTOMOC ON) # For meta object compiler
set(CMAKE_AUTORCC ON) # Resource files
set(CMAKE_AUTOUIC ON) # UI files

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 REQUIRED Widgets Gui ) 
ADD_DEFINITIONS(${QT_DEFINITIONS})
find_package(Qt5 REQUIRED COMPONENTS OpenGL Quick Core Gui Sql Charts)
#find_package(Qt5QuickCompiler)

set(QWT_INCLUDE_DIR /usr/local/qwt-6.2.0/src)
set ( QWT_INCLUDE_DIRS ${QWT_INCLUDE_DIR} )

# version
set ( _VERSION_FILE ${QWT_INCLUDE_DIR}/qwt_global.h )
if ( EXISTS ${_VERSION_FILE} )
  file ( STRINGS ${_VERSION_FILE} _VERSION_LINE REGEX "define[ ]+QWT_VERSION_STR" )
  if ( _VERSION_LINE )
    string ( REGEX REPLACE ".*define[ ]+QWT_VERSION_STR[ ]+\"([^\"]*)\".*" "\\1" QWT_VERSION_STRING "${_VERSION_LINE}" )
  endif ()
endif ()
unset ( _VERSION_FILE )

find_library ( QWT_LIBRARY
  NAMES qwt 
  HINTS /usr/local/qwt-6.2.0/lib
)
set ( QWT_LIBRARIES ${QWT_LIBRARY} )

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Qwt REQUIRED_VARS QWT_LIBRARY QWT_INCLUDE_DIR VERSION_VAR QWT_VERSION_STRING )

if (Qwt_FOUND AND NOT TARGET Qwt::Qwt)
  add_library(Qwt::Qwt UNKNOWN IMPORTED)
  set_target_properties(Qwt::Qwt PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${QWT_INCLUDE_DIRS}"
                        IMPORTED_LOCATION "${QWT_LIBRARIES}")
endif ()

mark_as_advanced (
  QWT_LIBRARY
  QWT_INCLUDE_DIR
)
include_directories(${QWT_INCLUDE_DIR})

find_package(PkgConfig)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

find_package(Threads)
find_package(Eigen3 REQUIRED)

set(VTK_DIR "~/VTK-7.0.1/build/lib/cmake")
find_package(VTK REQUIRED PATHS ${VTK_DIR}) 

find_package(OpenGL)
find_package(GLEW)

find_package (OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OPEN3D_DIR "~/Open3D/build/lib/cmake")
find_package(Open3D REQUIRED PATHS ${OPEN3D_DIR})

include_directories(src src/resource src/forms src/headers src/sources)

set(HEADERS
    src/headers/metric2DSS.h
    src/headers/metric2D.h
    src/headers/metric3D.h
    src/headers/mode2DSS.h
    src/headers/mode2DOD.h
    src/headers/mode3DOD.h
    src/headers/vtkWidget.h
    src/headers/mainwindow.h 
)

set(SOURCE_qedic
    src/sources/metric2DSS.cpp
    src/sources/metric2D.cpp
    src/sources/metric3D.cpp
    src/sources/mode2DSS.cpp
    src/sources/mode2DOD.cpp
    src/sources/mode3DOD.cpp
    src/sources/vtkWidget.cpp
    src/sources/mainwindow.cpp
    src/sources/main.cpp 
)

qt5_wrap_cpp(HEADER_qedic ${HEADERS})
qt5_wrap_ui(UI_qedic src/forms/mainwindow.ui) 
qt5_add_resources(RESOURCES src/resource/resources.qrc)
#qtquick_compiler_add_resources(RESOURCES src/resource/resources.qrc)

add_executable(qedic 
    ${SOURCE_qedic} ${HEADER_qedic} ${UI_qedic} ${RESOURCES}
    ${QWT_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS}
)


target_link_libraries(qedic ${OpenCV_LIBRARIES} 
    opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
    Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Charts Qt5::OpenGL
    ${QWT_LIBRARIES} ${VTK_LIBRARIES} Open3D::Open3D
    OpenGL::GL GLEW::GLEW
)


include(${VTK_USE_FILE})