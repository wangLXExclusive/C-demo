# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\clion\CLion 2020.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\clion\CLion 2020.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\c++Project\thread

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\c++Project\thread\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/alg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/alg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/alg.dir/flags.make

CMakeFiles/alg.dir/main.cpp.obj: CMakeFiles/alg.dir/flags.make
CMakeFiles/alg.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\c++Project\thread\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/alg.dir/main.cpp.obj"
	D:\Mingw\mingw32\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\alg.dir\main.cpp.obj -c E:\c++Project\thread\main.cpp

CMakeFiles/alg.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alg.dir/main.cpp.i"
	D:\Mingw\mingw32\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\c++Project\thread\main.cpp > CMakeFiles\alg.dir\main.cpp.i

CMakeFiles/alg.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alg.dir/main.cpp.s"
	D:\Mingw\mingw32\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\c++Project\thread\main.cpp -o CMakeFiles\alg.dir\main.cpp.s

# Object files for target alg
alg_OBJECTS = \
"CMakeFiles/alg.dir/main.cpp.obj"

# External object files for target alg
alg_EXTERNAL_OBJECTS =

alg.exe: CMakeFiles/alg.dir/main.cpp.obj
alg.exe: CMakeFiles/alg.dir/build.make
alg.exe: CMakeFiles/alg.dir/linklibs.rsp
alg.exe: CMakeFiles/alg.dir/objects1.rsp
alg.exe: CMakeFiles/alg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\c++Project\thread\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable alg.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\alg.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/alg.dir/build: alg.exe

.PHONY : CMakeFiles/alg.dir/build

CMakeFiles/alg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\alg.dir\cmake_clean.cmake
.PHONY : CMakeFiles/alg.dir/clean

CMakeFiles/alg.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\c++Project\thread E:\c++Project\thread E:\c++Project\thread\cmake-build-debug E:\c++Project\thread\cmake-build-debug E:\c++Project\thread\cmake-build-debug\CMakeFiles\alg.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/alg.dir/depend
