# - Tools for building GPU C files: libraries and build dependencies.
# This script locates the NVIDIA GPU C tools. It should work on linux, windows,
# and mac and should be reasonably up to date with GPU C releases.
#
# This script makes use of the standard find_package arguments of <VERSION>,
# REQUIRED and QUIET.  GPU_FOUND will report if an acceptable version of GPU
# was found.
#
# The script will prompt the user to specify GPU_TOOLKIT_ROOT_DIR if the prefix
# cannot be determined by the location of nvcc in the system path and REQUIRED
# is specified to find_package(). To use a different installed version of the
# toolkit set the environment variable GPU_BIN_PATH before running cmake
# (e.g. GPU_BIN_PATH=/usr/local/gpu1.0 instead of the default /usr/local/gpu)
# or set GPU_TOOLKIT_ROOT_DIR after configuring.  If you change the value of
# GPU_TOOLKIT_ROOT_DIR, various components that depend on the path will be
# relocated.
#
# It might be necessary to set GPU_TOOLKIT_ROOT_DIR manually on certain
# platforms, or to use a gpu runtime not installed in the default location. In
# newer versions of the toolkit the gpu library is included with the graphics
# driver- be sure that the driver version matches what is needed by the gpu
# runtime version.
#
# The following variables affect the behavior of the macros in the script (in
# alphebetical order).  Note that any of these flags can be changed multiple
# times in the same directory before calling GPU_ADD_EXECUTABLE,
# GPU_ADD_LIBRARY, GPU_COMPILE, GPU_COMPILE_PTX or GPU_WRAP_SRCS.
#
#  GPU_64_BIT_DEVICE_CODE (Default matches host bit size)
#  -- Set to ON to compile for 64 bit device code, OFF for 32 bit device code.
#     Note that making this different from the host code when generating object
#     or C files from GPU code just won't work, because size_t gets defined by
#     nvcc in the generated source.  If you compile to PTX and then load the
#     file yourself, you can mix bit sizes between device and host.
#
#  GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE (Default ON)
#  -- Set to ON if you want the custom build rule to be attached to the source
#     file in Visual Studio.  Turn OFF if you add the same gpu file to multiple
#     targets.
#
#     This allows the user to build the target from the GPU file; however, bad
#     things can happen if the GPU source file is added to multiple targets.
#     When performing parallel builds it is possible for the custom build
#     command to be run more than once and in parallel causing cryptic build
#     errors.  VS runs the rules for every source file in the target, and a
#     source can have only one rule no matter how many projects it is added to.
#     When the rule is run from multiple targets race conditions can occur on
#     the generated file.  Eventually everything will get built, but if the user
#     is unaware of this behavior, there may be confusion.  It would be nice if
#     this script could detect the reuse of source files across multiple targets
#     and turn the option off for the user, but no good solution could be found.
#
#  GPU_BUILD_CUBIN (Default OFF)
#  -- Set to ON to enable and extra compilation pass with the -cubin option in
#     Device mode. The output is parsed and register, shared memory usage is
#     printed during build.
#
#  GPU_BUILD_EMULATION (Default OFF for device mode)
#  -- Set to ON for Emulation mode. -D_DEVICEEMU is defined for GPU C files
#     when GPU_BUILD_EMULATION is TRUE.
#
#  GPU_GENERATED_OUTPUT_DIR (Default CMAKE_CURRENT_BINARY_DIR)
#  -- Set to the path you wish to have the generated files placed.  If it is
#     blank output files will be placed in CMAKE_CURRENT_BINARY_DIR.
#     Intermediate files will always be placed in
#     CMAKE_CURRENT_BINARY_DIR/CMakeFiles.
#
#  GPU_HOST_COMPILATION_CPP (Default ON)
#  -- Set to OFF for C compilation of host code.
#
#  GPU_NVCC_FLAGS
#  GPU_NVCC_FLAGS_<CONFIG>
#  -- Additional NVCC command line arguments.  NOTE: multiple arguments must be
#     semi-colon delimited (e.g. --compiler-options;-Wall)
#
#  GPU_PROPAGATE_HOST_FLAGS (Default ON)
#  -- Set to ON to propagate CMAKE_{C,CXX}_FLAGS and their configuration
#     dependent counterparts (e.g. CMAKE_C_FLAGS_DEBUG) automatically to the
#     host compiler through nvcc's -Xcompiler flag.  This helps make the
#     generated host code match the rest of the system better.  Sometimes
#     certain flags give nvcc problems, and this will help you turn the flag
#     propagation off.  This does not affect the flags supplied directly to nvcc
#     via GPU_NVCC_FLAGS or through the OPTION flags specified through
#     GPU_ADD_LIBRARY, GPU_ADD_EXECUTABLE, or GPU_WRAP_SRCS.  Flags used for
#     shared library compilation are not affected by this flag.
#
#  GPU_VERBOSE_BUILD (Default OFF)
#  -- Set to ON to see all the commands used when building the GPU file.  When
#     using a Makefile generator the value defaults to VERBOSE (run make
#     VERBOSE=1 to see output), although setting GPU_VERBOSE_BUILD to ON will
#     always print the output.
#
# The script creates the following macros (in alphebetical order):
#
#  GPU_ADD_CUFFT_TO_TARGET( gpu_target )
#  -- Adds the cufft library to the target (can be any target).  Handles whether
#     you are in emulation mode or not.
#
#  GPU_ADD_CUBLAS_TO_TARGET( gpu_target )
#  -- Adds the cublas library to the target (can be any target).  Handles
#     whether you are in emulation mode or not.
#
#  GPU_ADD_EXECUTABLE( gpu_target file0 file1 ...
#                       [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#  -- Creates an executable "gpu_target" which is made up of the files
#     specified.  All of the non GPU C files are compiled using the standard
#     build rules specified by CMAKE and the gpu files are compiled to object
#     files using nvcc and the host compiler.  In addition GPU_INCLUDE_DIRS is
#     added automatically to include_directories().  Some standard CMake target
#     calls can be used on the target after calling this macro
#     (e.g. set_target_properties and target_link_libraries), but setting
#     properties that adjust compilation flags will not affect code compiled by
#     nvcc.  Such flags should be modified before calling GPU_ADD_EXECUTABLE,
#     GPU_ADD_LIBRARY or GPU_WRAP_SRCS.
#
#  GPU_ADD_LIBRARY( gpu_target file0 file1 ...
#                    [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#  -- Same as GPU_ADD_EXECUTABLE except that a library is created.
#
#  GPU_BUILD_CLEAN_TARGET()
#  -- Creates a convience target that deletes all the dependency files
#     generated.  You should make clean after running this target to ensure the
#     dependency files get regenerated.
#
#  GPU_COMPILE( generated_files file0 file1 ... [STATIC | SHARED | MODULE]
#                [OPTIONS ...] )
#  -- Returns a list of generated files from the input source files to be used
#     with ADD_LIBRARY or ADD_EXECUTABLE.
#
#  GPU_COMPILE_PTX( generated_files file0 file1 ... [OPTIONS ...] )
#  -- Returns a list of PTX files generated from the input source files.
#
#  GPU_INCLUDE_DIRECTORIES( path0 path1 ... )
#  -- Sets the directories that should be passed to nvcc
#     (e.g. nvcc -Ipath0 -Ipath1 ... ). These paths usually contain other .cu
#     files.
#
#  GPU_WRAP_SRCS ( gpu_target format generated_files file0 file1 ...
#                   [STATIC | SHARED | MODULE] [OPTIONS ...] )
#  -- This is where all the magic happens.  GPU_ADD_EXECUTABLE,
#     GPU_ADD_LIBRARY, GPU_COMPILE, and GPU_COMPILE_PTX all call this
#     function under the hood.
#
#     Given the list of files (file0 file1 ... fileN) this macro generates
#     custom commands that generate either PTX or linkable objects (use "PTX" or
#     "OBJ" for the format argument to switch).  Files that don't end with .cu
#     or have the HEADER_FILE_ONLY property are ignored.
#
#     The arguments passed in after OPTIONS are extra command line options to
#     give to nvcc.  You can also specify per configuration options by
#     specifying the name of the configuration followed by the options.  General
#     options must preceed configuration specific options.  Not all
#     configurations need to be specified, only the ones provided will be used.
#
#        OPTIONS -DFLAG=2 "-DFLAG_OTHER=space in flag"
#        DEBUG -g
#        RELEASE --use_fast_math
#        RELWITHDEBINFO --use_fast_math;-g
#        MINSIZEREL --use_fast_math
#
#     For certain configurations (namely VS generating object files with
#     GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE set to ON), no generated file will
#     be produced for the given gpu file.  This is because when you add the
#     gpu file to Visual Studio it knows that this file produces an object file
#     and will link in the resulting object file automatically.
#
#     This script will also generate a separate cmake script that is used at
#     build time to invoke nvcc.  This is for several reasons.
#
#       1. nvcc can return negative numbers as return values which confuses
#       Visual Studio into thinking that the command succeeded.  The script now
#       checks the error codes and produces errors when there was a problem.
#
#       2. nvcc has been known to not delete incomplete results when it
#       encounters problems.  This confuses build systems into thinking the
#       target was generated when in fact an unusable file exists.  The script
#       now deletes the output files if there was an error.
#
#       3. By putting all the options that affect the build into a file and then
#       make the build rule dependent on the file, the output files will be
#       regenerated when the options change.
#
#     This script also looks at optional arguments STATIC, SHARED, or MODULE to
#     determine when to target the object compilation for a shared library.
#     BUILD_SHARED_LIBS is ignored in GPU_WRAP_SRCS, but it is respected in
#     GPU_ADD_LIBRARY.  On some systems special flags are added for building
#     objects intended for shared libraries.  A preprocessor macro,
#     <target_name>_EXPORTS is defined when a shared library compilation is
#     detected.
#
#     Flags passed into add_definitions with -D or /D are passed along to nvcc.
#
# The script defines the following variables:
#
#  GPU_VERSION_MAJOR    -- The major version of gpu as reported by nvcc.
#  GPU_VERSION_MINOR    -- The minor version.
#  GPU_VERSION
#  GPU_VERSION_STRING   -- GPU_VERSION_MAJOR.GPU_VERSION_MINOR
#
#  GPU_TOOLKIT_ROOT_DIR -- Path to the GPU Toolkit (defined if not set).
#  GPU_SDK_ROOT_DIR     -- Path to the GPU SDK.  Use this to find files in the
#                           SDK.  This script will not directly support finding
#                           specific libraries or headers, as that isn't
#                           supported by NVIDIA.  If you want to change
#                           libraries when the path changes see the
#                           FindGPU.cmake script for an example of how to clear
#                           these variables.  There are also examples of how to
#                           use the GPU_SDK_ROOT_DIR to locate headers or
#                           libraries, if you so choose (at your own risk).
#  GPU_INCLUDE_DIRS     -- Include directory for gpu headers.  Added automatically
#                           for GPU_ADD_EXECUTABLE and GPU_ADD_LIBRARY.
#  GPU_LIBRARIES        -- GPU RT library.
#  GPU_CUFFT_LIBRARIES  -- Device or emulation library for the GPU FFT
#                           implementation (alternative to:
#                           GPU_ADD_CUFFT_TO_TARGET macro)
#  GPU_CUBLAS_LIBRARIES -- Device or emulation library for the GPU BLAS
#                           implementation (alterative to:
#                           GPU_ADD_CUBLAS_TO_TARGET macro).
#  GPU_curand_LIBRARY   -- GPU Random Number Generation library.
#                           Only available for GPU version 3.2+.
#  GPU_cusparse_LIBRARY -- GPU Sparse Matrix library.
#                           Only available for GPU version 3.2+.
#  GPU_npp_LIBRARY      -- NVIDIA Performance Primitives library.
#                           Only available for GPU version 4.0+.
#  GPU_nvcuvenc_LIBRARY -- GPU Video Encoder library.
#                           Only available for GPU version 3.2+.
#                           Windows only.
#  GPU_nvcuvid_LIBRARY  -- GPU Video Decoder library.
#                           Only available for GPU version 3.2+.
#                           Windows only.
#
#
#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindGPU.html
#
#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#  Copyright (c) 2007-2009
#  Scientific Computing and Imaging Institute, University of Utah
#
#  This code is licensed under the MIT License.  See the FindGPU.cmake script
#  for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindGPU.cmake

# We need to have at least this version to support the VERSION_LESS argument to 'if' (2.6.2) and unset (2.6.3)
cmake_policy(PUSH)
cmake_minimum_required(VERSION 2.6.3)
cmake_policy(POP)

# This macro helps us find the location of helper files we will need the full path to
macro(GPU_FIND_HELPER_FILE _name _extension)
  set(_full_name "${_name}.${_extension}")
  # CMAKE_CURRENT_LIST_FILE contains the full path to the file currently being
  # processed.  Using this variable, we can pull out the current path, and
  # provide a way to get access to the other files we need local to here.
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  set(GPU_${_name} "${CMAKE_CURRENT_LIST_DIR}/FindGPU/${_full_name}")
  if(NOT EXISTS "${GPU_${_name}}")
    set(error_message "${_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}/FindGPU")
    if(GPU_FIND_REQUIRED)
      message(FATAL_ERROR "${error_message}")
    else()
      if(NOT GPU_FIND_QUIETLY)
        message(STATUS "${error_message}")
      endif()
    endif()
  endif()
  # Set this variable as internal, so the user isn't bugged with it.
  set(GPU_${_name} ${GPU_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro(GPU_FIND_HELPER_FILE)

#####################################################################
## GPU_INCLUDE_NVCC_DEPENDENCIES
##

# So we want to try and include the dependency file if it exists.  If
# it doesn't exist then we need to create an empty one, so we can
# include it.

# If it does exist, then we need to check to see if all the files it
# depends on exist.  If they don't then we should clear the dependency
# file and regenerate it later.  This covers the case where a header
# file has disappeared or moved.

macro(GPU_INCLUDE_NVCC_DEPENDENCIES dependency_file)
  set(GPU_NVCC_DEPEND)
  set(GPU_NVCC_DEPEND_REGENERATE FALSE)


  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindGPU.cmake generated file.  Do not edit.\n")
  endif()
  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
#   if(DEFINED GPU_NVCC_DEPEND)
#     message("GPU_NVCC_DEPEND set")
#   else()
#     message("GPU_NVCC_DEPEND NOT set")
#   endif()
  if(GPU_NVCC_DEPEND)
    #message("GPU_NVCC_DEPEND found")
    foreach(f ${GPU_NVCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(GPU_NVCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach(f)
  else(GPU_NVCC_DEPEND)
    #message("GPU_NVCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(GPU_NVCC_DEPEND_REGENERATE TRUE)
  endif(GPU_NVCC_DEPEND)

  #message("GPU_NVCC_DEPEND_REGENERATE = ${GPU_NVCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(GPU_NVCC_DEPEND_REGENERATE)
    set(GPU_NVCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindGPU.cmake generated file.  Do not edit.\n")
  endif(GPU_NVCC_DEPEND_REGENERATE)

endmacro(GPU_INCLUDE_NVCC_DEPENDENCIES)

###############################################################################
###############################################################################
# Setup variables' defaults
###############################################################################
###############################################################################

# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(GPU_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(GPU_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(GPU_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${GPU_64_BIT_DEVICE_CODE_DEFAULT})

# Attach the build rule to the source file in VS.  This option
option(GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE "Attach the build rule to the GPU source file.  Enable only when the GPU source file is added to at most one target." ON)

# Prints out extra information about the gpu file during compilation
option(GPU_BUILD_CUBIN "Generate and parse .cubin files in Device mode." OFF)

# Set whether we are using emulation or device mode.
option(GPU_BUILD_EMULATION "Build in Emulation mode" OFF)

# Where to put the generated output.
set(GPU_GENERATED_OUTPUT_DIR "" CACHE PATH "Directory to put all the output files.  If blank it will default to the CMAKE_CURRENT_BINARY_DIR")

# Parse HOST_COMPILATION mode.
option(GPU_HOST_COMPILATION_CPP "Generated file extension" ON)

# Extra user settable flags
set(GPU_NVCC_FLAGS "" CACHE STRING "Semi-colon delimit multiple arguments.")

# Propagate the host flags to the host compiler via -Xcompiler
option(GPU_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)

# Specifies whether the commands used when compiling the .cu file will be printed out.
option(GPU_VERBOSE_BUILD "Print out the commands run while compiling the GPU source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)

mark_as_advanced(
  GPU_64_BIT_DEVICE_CODE
  GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE
  GPU_GENERATED_OUTPUT_DIR
  GPU_HOST_COMPILATION_CPP
  GPU_NVCC_FLAGS
  GPU_PROPAGATE_HOST_FLAGS
  )

# Makefile and similar generators don't define CMAKE_CONFIGURATION_TYPES, so we
# need to add another entry for the CMAKE_BUILD_TYPE.  We also need to add the
# standerd set of 4 build types (Debug, MinSizeRel, Release, and RelWithDebInfo)
# for completeness.  We need run this loop in order to accomodate the addition
# of extra configuration types.  Duplicate entries will be removed by
# REMOVE_DUPLICATES.
set(GPU_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES GPU_configuration_types)
foreach(config ${GPU_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(GPU_NVCC_FLAGS_${config_upper} "" CACHE STRING "Semi-colon delimit multiple arguments.")
    mark_as_advanced(GPU_NVCC_FLAGS_${config_upper})
endforeach()

###############################################################################
###############################################################################
# Locate GPU, Set Build Type, etc.
###############################################################################
###############################################################################

# Check to see if the GPU_TOOLKIT_ROOT_DIR and GPU_SDK_ROOT_DIR have changed,
# if they have then clear the cache variables, so that will be detected again.
if(NOT "${GPU_TOOLKIT_ROOT_DIR}" STREQUAL "${GPU_TOOLKIT_ROOT_DIR_INTERNAL}")
  unset(GPU_NVCC_EXECUTABLE CACHE)
  unset(GPU_TOOLKIT_INCLUDE CACHE)
  unset(GPU_GPURT_LIBRARY CACHE)
  # Make sure you run this before you unset GPU_VERSION.
  if(GPU_VERSION VERSION_EQUAL "3.0")
    # This only existed in the 3.0 version of the GPU toolkit
    unset(GPU_GPURTEMU_LIBRARY CACHE)
  endif()
  unset(GPU_VERSION CACHE)
  unset(GPU_GPU_LIBRARY CACHE)
  unset(GPU_cublas_LIBRARY CACHE)
  unset(GPU_cublasemu_LIBRARY CACHE)
  unset(GPU_cufft_LIBRARY CACHE)
  unset(GPU_cufftemu_LIBRARY CACHE)
  unset(GPU_curand_LIBRARY CACHE)
  unset(GPU_cusparse_LIBRARY CACHE)
  unset(GPU_npp_LIBRARY CACHE)
  unset(GPU_nvcuvenc_LIBRARY CACHE)
  unset(GPU_nvcuvid_LIBRARY CACHE)
endif()

if(NOT "${GPU_SDK_ROOT_DIR}" STREQUAL "${GPU_SDK_ROOT_DIR_INTERNAL}")
  # No specific variables to catch.  Use this kind of code before calling
  # find_package(GPU) to clean up any variables that may depend on this path.

  #   unset(MY_SPECIAL_GPU_SDK_INCLUDE_DIR CACHE)
  #   unset(MY_SPECIAL_GPU_SDK_LIBRARY CACHE)
endif()

# Search for the gpu distribution.
if(NOT GPU_TOOLKIT_ROOT_DIR)

  # Search in the GPU_BIN_PATH first.
  find_path(GPU_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS
      ENV GPU_PATH
      ENV GPU_BIN_PATH
    PATH_SUFFIXES bin bin64
    DOC "Toolkit location."
    NO_DEFAULT_PATH
    )
  # Now search default paths
  find_path(GPU_TOOLKIT_ROOT_DIR
    NAMES nvcc nvcc.exe
    PATHS /usr/local/bin
          /usr/local/gpu/bin
    DOC "Toolkit location."
    )

  if (GPU_TOOLKIT_ROOT_DIR)
    string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" GPU_TOOLKIT_ROOT_DIR ${GPU_TOOLKIT_ROOT_DIR})
    # We need to force this back into the cache.
    set(GPU_TOOLKIT_ROOT_DIR ${GPU_TOOLKIT_ROOT_DIR} CACHE PATH "Toolkit location." FORCE)
  endif(GPU_TOOLKIT_ROOT_DIR)
  if (NOT EXISTS ${GPU_TOOLKIT_ROOT_DIR})
    if(GPU_FIND_REQUIRED)
      message(FATAL_ERROR "Specify GPU_TOOLKIT_ROOT_DIR")
    elseif(NOT GPU_FIND_QUIETLY)
      message(STATUS "GPU_TOOLKIT_ROOT_DIR not found or specified")
    endif()
  endif (NOT EXISTS ${GPU_TOOLKIT_ROOT_DIR})
endif (NOT GPU_TOOLKIT_ROOT_DIR)

# GPU_NVCC_EXECUTABLE
find_program(GPU_NVCC_EXECUTABLE
  NAMES nvcc
  PATHS "${GPU_TOOLKIT_ROOT_DIR}"
  ENV GPU_PATH
  ENV GPU_BIN_PATH
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_program(GPU_NVCC_EXECUTABLE nvcc)
mark_as_advanced(GPU_NVCC_EXECUTABLE)

if(GPU_NVCC_EXECUTABLE AND NOT GPU_VERSION)
  # Compute the version.
  execute_process (COMMAND ${GPU_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" GPU_VERSION_MAJOR ${NVCC_OUT})
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" GPU_VERSION_MINOR ${NVCC_OUT})
  set(GPU_VERSION "${GPU_VERSION_MAJOR}.${GPU_VERSION_MINOR}" CACHE STRING "Version of GPU as computed from nvcc.")
  mark_as_advanced(GPU_VERSION)
else()
  # Need to set these based off of the cached value
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" GPU_VERSION_MAJOR "${GPU_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" GPU_VERSION_MINOR "${GPU_VERSION}")
endif()

# Always set this convenience variable
set(GPU_VERSION_STRING "${GPU_VERSION}")

# Here we need to determine if the version we found is acceptable.  We will
# assume that is unless GPU_FIND_VERSION_EXACT or GPU_FIND_VERSION is
# specified.  The presence of either of these options checks the version
# string and signals if the version is acceptable or not.
set(_gpu_version_acceptable TRUE)
#
if(GPU_FIND_VERSION_EXACT AND NOT GPU_VERSION VERSION_EQUAL GPU_FIND_VERSION)
  set(_gpu_version_acceptable FALSE)
endif()
#
if(GPU_FIND_VERSION AND GPU_VERSION VERSION_LESS  GPU_FIND_VERSION)
  set(_gpu_version_acceptable FALSE)
endif()
#
if(NOT _gpu_version_acceptable AND GPU_VERSION)
  set(_gpu_error_message "Requested GPU version ${GPU_FIND_VERSION}, but found unacceptable version ${GPU_VERSION}")
  if(GPU_FIND_REQUIRED)
    message(STATUS "${_gpu_error_message}")
  elseif(NOT GPU_FIND_QUIETLY)
    message(STATUS "${_gpu_error_message}")
  endif()
endif()

# GPU_TOOLKIT_INCLUDE
find_path(GPU_TOOLKIT_INCLUDE
  device_functions.h # Header included in toolkit
  PATHS "${GPU_TOOLKIT_ROOT_DIR}"
  ENV GPU_PATH
  ENV GPU_INC_PATH
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_path(GPU_TOOLKIT_INCLUDE device_functions.h)
mark_as_advanced(GPU_TOOLKIT_INCLUDE)

# Set the user list of include dir to nothing to initialize it.
set (GPU_NVCC_INCLUDE_ARGS_USER "")
set (GPU_INCLUDE_DIRS ${GPU_TOOLKIT_INCLUDE})

macro(FIND_LIBRARY_LOCAL_FIRST _var _names _doc)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # GPU 3.2+ on Windows moved the library directoryies, so we need the new
    # and old paths.
    set(_gpu_64bit_lib_dir "lib/x64" "lib64" )
  endif()
  # GPU 3.2+ on Windows moved the library directories, so we need to new
  # (lib/Win32) and the old path (lib).
  find_library(${_var}
    NAMES ${_names}
    PATHS "${GPU_TOOLKIT_ROOT_DIR}"
    ENV GPU_PATH
    ENV GPU_LIB_PATH
    PATH_SUFFIXES ${_gpu_64bit_lib_dir} "lib/Win32" "lib"
    DOC ${_doc}
    NO_DEFAULT_PATH
    )
  # Search default search paths, after we search our own set of paths.
  find_library(${_var} NAMES ${_names} DOC ${_doc})
endmacro()

# GPU_LIBRARIES
find_library_local_first(GPU_GPURT_LIBRARY gpurt "\"gpurt\" library")
if(GPU_VERSION VERSION_EQUAL "3.0")
  # The gpurtemu library only existed for the 3.0 version of GPU.
  find_library_local_first(GPU_GPURTEMU_LIBRARY gpurtemu "\"gpurtemu\" library")
  mark_as_advanced(
    GPU_GPURTEMU_LIBRARY
    )
endif()
# If we are using emulation mode and we found the gpurtemu library then use
# that one instead of gpurt.
if(GPU_BUILD_EMULATION AND GPU_GPURTEMU_LIBRARY)
  set(GPU_LIBRARIES ${GPU_GPURTEMU_LIBRARY})
else()
  set(GPU_LIBRARIES ${GPU_GPURT_LIBRARY})
endif()
if(APPLE)
  # We need to add the path to gpurt to the linker using rpath, since the
  # library name for the gpu libraries is prepended with @rpath.
  if(GPU_BUILD_EMULATION AND GPU_GPURTEMU_LIBRARY)
    get_filename_component(_gpu_path_to_gpurt "${GPU_GPURTEMU_LIBRARY}" PATH)
  else()
    get_filename_component(_gpu_path_to_gpurt "${GPU_GPURT_LIBRARY}" PATH)
  endif()
  if(_gpu_path_to_gpurt)
    list(APPEND GPU_LIBRARIES -Wl,-rpath "-Wl,${_gpu_path_to_gpurt}")
  endif()
endif()

# 1.1 toolkit on linux doesn't appear to have a separate library on
# some platforms.
find_library_local_first(GPU_GPU_LIBRARY gpu "\"gpu\" library (older versions only).")

# Add gpu library to the link line only if it is found.
if (GPU_GPU_LIBRARY)
  set(GPU_LIBRARIES ${GPU_LIBRARIES} ${GPU_GPU_LIBRARY})
endif(GPU_GPU_LIBRARY)

mark_as_advanced(
  GPU_GPU_LIBRARY
  GPU_GPURT_LIBRARY
  )

#######################
# Look for some of the toolkit helper libraries
macro(FIND_GPU_HELPER_LIBS _name)
  find_library_local_first(GPU_${_name}_LIBRARY ${_name} "\"${_name}\" library")
  mark_as_advanced(GPU_${_name}_LIBRARY)
endmacro(FIND_GPU_HELPER_LIBS)

#######################
# Disable emulation for v3.1 onward
if(GPU_VERSION VERSION_GREATER "3.0")
  if(GPU_BUILD_EMULATION)
    message(FATAL_ERROR "GPU_BUILD_EMULATION is not supported in version 3.1 and onwards.  You must disable it to proceed.  You have version ${GPU_VERSION}.")
  endif()
endif()

# Search for additional GPU toolkit libraries.
if(GPU_VERSION VERSION_LESS "3.1")
  # Emulation libraries aren't available in version 3.1 onward.
  find_gpu_helper_libs(cufftemu)
  find_gpu_helper_libs(cublasemu)
endif()
find_gpu_helper_libs(cufft)
find_gpu_helper_libs(cublas)
if(NOT GPU_VERSION VERSION_LESS "3.2")
  # cusparse showed up in version 3.2
  find_gpu_helper_libs(cusparse)
  find_gpu_helper_libs(curand)
  if (WIN32)
    find_gpu_helper_libs(nvcuvenc)
    find_gpu_helper_libs(nvcuvid)
  endif()
endif()
if(NOT GPU_VERSION VERSION_LESS "4.0")
  find_gpu_helper_libs(npp)
endif()

if (GPU_BUILD_EMULATION)
  set(GPU_CUFFT_LIBRARIES ${GPU_cufftemu_LIBRARY})
  set(GPU_CUBLAS_LIBRARIES ${GPU_cublasemu_LIBRARY})
else()
  set(GPU_CUFFT_LIBRARIES ${GPU_cufft_LIBRARY})
  set(GPU_CUBLAS_LIBRARIES ${GPU_cublas_LIBRARY})
endif()

########################
# Look for the SDK stuff.  As of GPU 3.0 NVSDKGPU_ROOT has been replaced with
# NVSDKCOMPUTE_ROOT with the old GPU C contents moved into the C subdirectory
find_path(GPU_SDK_ROOT_DIR common/inc/cutil.h
  "$ENV{NVSDKCOMPUTE_ROOT}/C"
  "$ENV{NVSDKGPU_ROOT}"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
  "/Developer/GPU\ Computing/C"
  )

# Keep the GPU_SDK_ROOT_DIR first in order to be able to override the
# environment variables.
set(GPU_SDK_SEARCH_PATH
  "${GPU_SDK_ROOT_DIR}"
  "${GPU_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
  "${GPU_TOOLKIT_ROOT_DIR}/NVSDK0.2"
  "${GPU_TOOLKIT_ROOT_DIR}/NV_GPU_SDK"
  "$ENV{HOME}/NVIDIA_GPU_SDK"
  "$ENV{HOME}/NVIDIA_GPU_SDK_MACOSX"
  "/Developer/GPU"
  )

# Example of how to find an include file from the GPU_SDK_ROOT_DIR

# find_path(GPU_CUT_INCLUDE_DIR
#   cutil.h
#   PATHS ${GPU_SDK_SEARCH_PATH}
#   PATH_SUFFIXES "common/inc"
#   DOC "Location of cutil.h"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_path(GPU_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

# mark_as_advanced(GPU_CUT_INCLUDE_DIR)


# Example of how to find a library in the GPU_SDK_ROOT_DIR

# # cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# # to get these confused, so we are setting the name based on the word size of
# # the build.

# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(gpu_cutil_name cutil64)
# else(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(gpu_cutil_name cutil32)
# endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

# find_library(GPU_CUT_LIBRARY
#   NAMES cutil ${gpu_cutil_name}
#   PATHS ${GPU_SDK_SEARCH_PATH}
#   # The new version of the sdk shows up in common/lib, but the old one is in lib
#   PATH_SUFFIXES "common/lib" "lib"
#   DOC "Location of cutil library"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_library(GPU_CUT_LIBRARY NAMES cutil ${gpu_cutil_name} DOC "Location of cutil library")
# mark_as_advanced(GPU_CUT_LIBRARY)
# set(GPU_CUT_LIBRARIES ${GPU_CUT_LIBRARY})



#############################
# Check for required components
set(GPU_FOUND TRUE)

set(GPU_TOOLKIT_ROOT_DIR_INTERNAL "${GPU_TOOLKIT_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time GPU_TOOLKIT_ROOT_DIR was set successfully." FORCE)
set(GPU_SDK_ROOT_DIR_INTERNAL "${GPU_SDK_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time GPU_SDK_ROOT_DIR was set successfully." FORCE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GPU DEFAULT_MSG
  GPU_TOOLKIT_ROOT_DIR
  GPU_NVCC_EXECUTABLE
  GPU_INCLUDE_DIRS
  GPU_GPURT_LIBRARY
  _gpu_version_acceptable
  )



###############################################################################
###############################################################################
# Macros
###############################################################################
###############################################################################

###############################################################################
# Add include directories to pass to the nvcc command.
macro(GPU_INCLUDE_DIRECTORIES)
  foreach(dir ${ARGN})
    list(APPEND GPU_NVCC_INCLUDE_ARGS_USER -I${dir})
  endforeach(dir ${ARGN})
endmacro(GPU_INCLUDE_DIRECTORIES)


##############################################################################
gpu_find_helper_file(parse_cubin cmake)
gpu_find_helper_file(make2cmake cmake)
gpu_find_helper_file(run_nvcc cmake)

##############################################################################
# Separate the OPTIONS out from the sources
#
macro(GPU_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if(arg STREQUAL "OPTIONS")
      set( _found_options TRUE )
    elseif(
        arg STREQUAL "WIN32" OR
        arg STREQUAL "MACOSX_BUNDLE" OR
        arg STREQUAL "EXCLUDE_FROM_ALL" OR
        arg STREQUAL "STATIC" OR
        arg STREQUAL "SHARED" OR
        arg STREQUAL "MODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

##############################################################################
# Parse the OPTIONS from ARGN and set the variables prefixed by _option_prefix
#
macro(GPU_PARSE_NVCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${GPU_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()

##############################################################################
# Helper to add the include directory for GPU only once
function(GPU_ADD_GPU_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${GPU_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(${GPU_INCLUDE_DIRS})
  endif()
endfunction()

function(GPU_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _gpu_found_SHARED)
  list(FIND cmake_args MODULE _gpu_found_MODULE)
  list(FIND cmake_args STATIC _gpu_found_STATIC)
  if( _gpu_found_SHARED GREATER -1 OR
      _gpu_found_MODULE GREATER -1 OR
      _gpu_found_STATIC GREATER -1)
    set(_gpu_build_shared_libs)
  else()
    if (BUILD_SHARED_LIBS)
      set(_gpu_build_shared_libs SHARED)
    else()
      set(_gpu_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_gpu_build_shared_libs} PARENT_SCOPE)
endfunction()

##############################################################################
# Helper to avoid clashes of files with the same basename but different paths.
# This doesn't attempt to do exactly what CMake internals do, which is to only
# add this path when there is a conflict, since by the time a second collision
# in names is detected it's already too late to fix the first one.  For
# consistency sake the relative path will be added to all files.
function(GPU_COMPUTE_BUILD_PATH path build_path)
  #message("GPU_COMPUTE_BUILD_PATH([${path}] ${build_path})")
  # Only deal with CMake style paths from here on out
  file(TO_CMAKE_PATH "${path}" bpath)
  if (IS_ABSOLUTE "${bpath}")
    # Absolute paths are generally unnessary, especially if something like
    # FILE(GLOB_RECURSE) is used to pick up the files.
    file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
  endif()

  # This recipie is from cmLocalGenerator::CreateSafeUniqueObjectFileName in the
  # CMake source.

  # Remove leading /
  string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
  # Avoid absolute paths by removing ':'
  string(REPLACE ":" "_" bpath "${bpath}")
  # Avoid relative paths that go up the tree
  string(REPLACE "../" "__/" bpath "${bpath}")
  # Avoid spaces
  string(REPLACE " " "_" bpath "${bpath}")

  # Strip off the filename.  I wait until here to do it, since removin the
  # basename can make a path that looked like path/../basename turn into
  # path/.. (notice the trailing slash).
  get_filename_component(bpath "${bpath}" PATH)

  set(${build_path} "${bpath}" PARENT_SCOPE)
  #message("${build_path} = ${bpath}")
endfunction()

##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to invoke the nvcc compiler to generate C or PTX source
# dependent upon the format parameter.  The compiler is invoked once with -M
# to generate a dependency file and a second time with -gpu or -ptx to generate
# a .cpp or .ptx file.
# INPUT:
#   gpu_target         - Target name
#   format              - PTX or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to NVCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
##############################################################################

macro(GPU_WRAP_SRCS gpu_target format generated_files)

  if( ${format} MATCHES "PTX" )
    set( compile_to_ptx ON )
  elseif( ${format} MATCHES "OBJ")
    set( compile_to_ptx OFF )
  else()
    message( FATAL_ERROR "Invalid format flag passed to GPU_WRAP_SRCS: '${format}'.  Use OBJ or PTX.")
  endif()

  # Set up all the command line flags here, so that they can be overriden on a per target basis.

  set(nvcc_flags "")

  # Emulation if the card isn't present.
  if (GPU_BUILD_EMULATION)
    # Emulation.
    set(nvcc_flags ${nvcc_flags} --device-emulation -D_DEVICEEMU -g)
  else(GPU_BUILD_EMULATION)
    # Device mode.  No flags necessary.
  endif(GPU_BUILD_EMULATION)

  if(GPU_HOST_COMPILATION_CPP)
    set(GPU_C_OR_CXX CXX)
  else(GPU_HOST_COMPILATION_CPP)
    if(GPU_VERSION VERSION_LESS "3.0")
      set(nvcc_flags ${nvcc_flags} --host-compilation C)
    else()
      message(WARNING "--host-compilation flag is deprecated in GPU version >= 3.0.  Removing --host-compilation C flag" )
    endif()
    set(GPU_C_OR_CXX C)
  endif(GPU_HOST_COMPILATION_CPP)

  set(generated_extension ${CMAKE_${GPU_C_OR_CXX}_OUTPUT_EXTENSION})

  if(GPU_64_BIT_DEVICE_CODE)
    set(nvcc_flags ${nvcc_flags} -m64)
  else()
    set(nvcc_flags ${nvcc_flags} -m32)
  endif()

  # This needs to be passed in at this stage, because VS needs to fill out the
  # value of VCInstallDir from within VS.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      # Add nvcc flag for 64b Windows
      set(ccbin_flags -D "\"CCBIN:PATH=$(VCInstallDir)bin\"" )
    endif()
  endif()

  # Figure out which configure we will use and pass that in as an argument to
  # the script.  We need to defer the decision until compilation time, because
  # for VS projects we won't know if we are making a debug or release build
  # until build time.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set( GPU_build_configuration "$(ConfigurationName)" )
  else()
    set( GPU_build_configuration "${CMAKE_BUILD_TYPE}")
  endif()

  # Initialize our list of includes with the user ones followed by the GPU system ones.
  set(GPU_NVCC_INCLUDE_ARGS ${GPU_NVCC_INCLUDE_ARGS_USER} "-I${GPU_INCLUDE_DIRS}")
  # Get the include directories for this directory and use them for our nvcc command.
  get_directory_property(GPU_NVCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
  if(GPU_NVCC_INCLUDE_DIRECTORIES)
    foreach(dir ${GPU_NVCC_INCLUDE_DIRECTORIES})
      list(APPEND GPU_NVCC_INCLUDE_ARGS -I${dir})
    endforeach()
  endif()

  # Reset these variables
  set(GPU_WRAP_OPTION_NVCC_FLAGS)
  foreach(config ${GPU_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(GPU_WRAP_OPTION_NVCC_FLAGS_${config_upper})
  endforeach()

  GPU_GET_SOURCES_AND_OPTIONS(_gpu_wrap_sources _gpu_wrap_cmake_options _gpu_wrap_options ${ARGN})
  GPU_PARSE_NVCC_OPTIONS(GPU_WRAP_OPTION_NVCC_FLAGS ${_gpu_wrap_options})

  # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
  # respected in GPU_ADD_LIBRARY.
  set(_gpu_build_shared_libs FALSE)
  # SHARED, MODULE
  list(FIND _gpu_wrap_cmake_options SHARED _gpu_found_SHARED)
  list(FIND _gpu_wrap_cmake_options MODULE _gpu_found_MODULE)
  if(_gpu_found_SHARED GREATER -1 OR _gpu_found_MODULE GREATER -1)
    set(_gpu_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _gpu_wrap_cmake_options STATIC _gpu_found_STATIC)
  if(_gpu_found_STATIC GREATER -1)
    set(_gpu_build_shared_libs FALSE)
  endif()

  # GPU_HOST_FLAGS
  if(_gpu_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(GPU_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${GPU_C_OR_CXX}_FLAGS})
  else()
    set(GPU_HOST_SHARED_FLAGS)
  endif()
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  if(GPU_PROPAGATE_HOST_FLAGS)
    set(GPU_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${CMAKE_${GPU_C_OR_CXX}_FLAGS} ${GPU_HOST_SHARED_FLAGS})")
  else()
    set(GPU_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${GPU_HOST_SHARED_FLAGS})")
  endif()

  set(GPU_NVCC_FLAGS_CONFIG "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_nvcc.cmake
  foreach(config ${GPU_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we convert the strings to lists (like we want).

    if(GPU_PROPAGATE_HOST_FLAGS)
      # nvcc chokes on -g3 in versions previous to 3.0, so replace it with -g
      if(CMAKE_COMPILER_IS_GNUCC AND GPU_VERSION VERSION_LESS "3.0")
        string(REPLACE "-g3" "-g" _gpu_C_FLAGS "${CMAKE_${GPU_C_OR_CXX}_FLAGS_${config_upper}}")
      else()
        set(_gpu_C_FLAGS "${CMAKE_${GPU_C_OR_CXX}_FLAGS_${config_upper}}")
      endif()

      set(GPU_HOST_FLAGS "${GPU_HOST_FLAGS}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_gpu_C_FLAGS})")
    endif()

    # Note that if we ever want GPU_NVCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is currently), we can remove the quotes around the
    # ${GPU_NVCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(GPU_NVCC_FLAGS_CONFIG "${GPU_NVCC_FLAGS_CONFIG}\nset(GPU_NVCC_FLAGS_${config_upper} ${GPU_NVCC_FLAGS_${config_upper}} ;; ${GPU_WRAP_OPTION_NVCC_FLAGS_${config_upper}})")
  endforeach()

  if(compile_to_ptx)
    # Don't use any of the host compilation flags for PTX targets.
    set(GPU_HOST_FLAGS)
    set(GPU_NVCC_FLAGS_CONFIG)
  endif()

  # Get the list of definitions from the directory property
  get_directory_property(GPU_NVCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(GPU_NVCC_DEFINITIONS)
    foreach(_definition ${GPU_NVCC_DEFINITIONS})
      list(APPEND nvcc_flags "-D${_definition}")
    endforeach()
  endif()

  if(_gpu_build_shared_libs)
    list(APPEND nvcc_flags "-D${gpu_target}_EXPORTS")
  endif()

  # Reset the output variable
  set(_gpu_wrap_generated_files "")

  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  foreach(file ${ARGN})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    if(${file} MATCHES ".*\\.cu$" AND NOT _is_header)

      # Determine output directory
      gpu_compute_build_path("${file}" gpu_build_path)
      set(gpu_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${gpu_target}.dir/${gpu_build_path}")
      if(GPU_GENERATED_OUTPUT_DIR)
        set(gpu_compile_output_dir "${GPU_GENERATED_OUTPUT_DIR}")
      else()
        if ( compile_to_ptx )
          set(gpu_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
        else()
          set(gpu_compile_output_dir "${gpu_compile_intermediate_directory}")
        endif()
      endif()

      # Add a custom target to generate a c or ptx file. ######################

      get_filename_component( basename ${file} NAME )
      if( compile_to_ptx )
        set(generated_file_path "${gpu_compile_output_dir}")
        set(generated_file_basename "${gpu_target}_generated_${basename}.ptx")
        set(format_flag "-ptx")
        file(MAKE_DIRECTORY "${gpu_compile_output_dir}")
      else( compile_to_ptx )
        set(generated_file_path "${gpu_compile_output_dir}/${CMAKE_CFG_INTDIR}")
        set(generated_file_basename "${gpu_target}_generated_${basename}${generated_extension}")
        set(format_flag "-c")
      endif( compile_to_ptx )

      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${gpu_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(NVCC_generated_dependency_file "${gpu_compile_intermediate_directory}/${generated_file_basename}.NVCC-depend")
      set(generated_cubin_file "${generated_file_path}/${generated_file_basename}.cubin.txt")
      set(custom_target_script "${gpu_compile_intermediate_directory}/${generated_file_basename}.cmake")

      # Setup properties for obj files:
      if( NOT compile_to_ptx )
        set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
      endif()

      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()

      # Bring in the dependencies.  Creates a variable GPU_NVCC_DEPEND #######
      gpu_include_nvcc_dependencies(${cmake_dependency_file})

      # Convience string for output ###########################################
      if(GPU_BUILD_EMULATION)
        set(gpu_build_type "Emulation")
      else(GPU_BUILD_EMULATION)
        set(gpu_build_type "Device")
      endif(GPU_BUILD_EMULATION)

      # Build the NVCC made dependency file ###################################
      set(build_cubin OFF)
      if ( NOT GPU_BUILD_EMULATION AND GPU_BUILD_CUBIN )
         if ( NOT compile_to_ptx )
           set ( build_cubin ON )
         endif( NOT compile_to_ptx )
      endif( NOT GPU_BUILD_EMULATION AND GPU_BUILD_CUBIN )

      # Configure the build script
      configure_file("${GPU_run_nvcc}" "${custom_target_script}" @ONLY)

      # So if a user specifies the same gpu file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      if(GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE)
        set(main_dep MAIN_DEPENDENCY ${source_file})
      else()
        set(main_dep DEPENDS ${source_file})
      endif()

      if(GPU_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()

      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      if(compile_to_ptx)
        set(gpu_build_comment_string "Building NVCC ptx file ${generated_file_relative_path}")
      else()
        set(gpu_build_comment_string "Building NVCC (${gpu_build_type}) object ${generated_file_relative_path}")
      endif()

      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${GPU_NVCC_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${GPU_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_cubin_file:STRING=${generated_cubin_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${gpu_compile_intermediate_directory}"
        COMMENT "${gpu_build_comment_string}"
        )

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      # Don't add the object file to the list of generated files if we are using
      # visual studio and we are attaching the build rule to the gpu file.  VS
      # will add our object file to the linker automatically for us.
      set(gpu_add_generated_file TRUE)

      if(NOT compile_to_ptx AND CMAKE_GENERATOR MATCHES "Visual Studio" AND GPU_ATTACH_VS_BUILD_RULE_TO_GPU_FILE)
        # Visual Studio 8 crashes when you close the solution when you don't add the object file.
        if(NOT CMAKE_GENERATOR MATCHES "Visual Studio 8")
          #message("Not adding ${generated_file}")
          set(gpu_add_generated_file FALSE)
        endif()
      endif()

      if(gpu_add_generated_file)
        list(APPEND _gpu_wrap_generated_files ${generated_file})
      endif()

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND GPU_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES GPU_ADDITIONAL_CLEAN_FILES)
      set(GPU_ADDITIONAL_CLEAN_FILES ${GPU_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the gpu dependency scanning.")

    endif(${file} MATCHES ".*\\.cu$" AND NOT _is_header)
  endforeach(file)

  # Set the return parameter
  set(${generated_files} ${_gpu_wrap_generated_files})
endmacro(GPU_WRAP_SRCS)


###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(GPU_ADD_LIBRARY gpu_target)

  GPU_ADD_GPU_INCLUDE_ONCE()

  # Separate the sources from the options
  GPU_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  GPU_BUILD_SHARED_LIBRARY(_gpu_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  GPU_WRAP_SRCS( ${gpu_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_gpu_shared_flag}
    OPTIONS ${_options} )

  # Add the library.
  add_library(${gpu_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    )

  target_link_libraries(${gpu_target}
    ${GPU_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. GPU_C_OR_CXX is computed based on GPU_HOST_COMPILATION_CPP.
  set_target_properties(${gpu_target}
    PROPERTIES
    LINKER_LANGUAGE ${GPU_C_OR_CXX}
    )

endmacro(GPU_ADD_LIBRARY gpu_target)


###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
macro(GPU_ADD_EXECUTABLE gpu_target)

  GPU_ADD_GPU_INCLUDE_ONCE()

  # Separate the sources from the options
  GPU_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  GPU_WRAP_SRCS( ${gpu_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Add the library.
  add_executable(${gpu_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    )

  target_link_libraries(${gpu_target}
    ${GPU_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. GPU_C_OR_CXX is computed based on GPU_HOST_COMPILATION_CPP.
  set_target_properties(${gpu_target}
    PROPERTIES
    LINKER_LANGUAGE ${GPU_C_OR_CXX}
    )

endmacro(GPU_ADD_EXECUTABLE gpu_target)


###############################################################################
###############################################################################
# GPU COMPILE
###############################################################################
###############################################################################
macro(GPU_COMPILE generated_files)

  # Separate the sources from the options
  GPU_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  GPU_WRAP_SRCS( gpu_compile OBJ _generated_files ${_sources} ${_cmake_options}
    OPTIONS ${_options} )

  set( ${generated_files} ${_generated_files})

endmacro(GPU_COMPILE)


###############################################################################
###############################################################################
# GPU COMPILE PTX
###############################################################################
###############################################################################
macro(GPU_COMPILE_PTX generated_files)

  # Separate the sources from the options
  GPU_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  GPU_WRAP_SRCS( gpu_compile_ptx PTX _generated_files ${_sources} ${_cmake_options}
    OPTIONS ${_options} )

  set( ${generated_files} ${_generated_files})

endmacro(GPU_COMPILE_PTX)

###############################################################################
###############################################################################
# GPU ADD CUFFT TO TARGET
###############################################################################
###############################################################################
macro(GPU_ADD_CUFFT_TO_TARGET target)
  if (GPU_BUILD_EMULATION)
    target_link_libraries(${target} ${GPU_cufftemu_LIBRARY})
  else()
    target_link_libraries(${target} ${GPU_cufft_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# GPU ADD CUBLAS TO TARGET
###############################################################################
###############################################################################
macro(GPU_ADD_CUBLAS_TO_TARGET target)
  if (GPU_BUILD_EMULATION)
    target_link_libraries(${target} ${GPU_cublasemu_LIBRARY})
  else()
    target_link_libraries(${target} ${GPU_cublas_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# GPU BUILD CLEAN TARGET
###############################################################################
###############################################################################
macro(GPU_BUILD_CLEAN_TARGET)
  # Call this after you add all your GPU targets, and you will get a convience
  # target.  You should also make clean after running this target to get the
  # build system to generate all the code again.

  set(gpu_clean_target_name clean_gpu_depends)
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOUPPER ${gpu_clean_target_name} gpu_clean_target_name)
  endif()
  add_custom_target(${gpu_clean_target_name}
    COMMAND ${CMAKE_COMMAND} -E remove ${GPU_ADDITIONAL_CLEAN_FILES})

  # Clear out the variable, so the next time we configure it will be empty.
  # This is useful so that the files won't persist in the list after targets
  # have been removed.
  set(GPU_ADDITIONAL_CLEAN_FILES "" CACHE INTERNAL "List of intermediate files that are part of the gpu dependency scanning.")
endmacro(GPU_BUILD_CLEAN_TARGET)
