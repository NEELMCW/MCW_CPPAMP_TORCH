# ** C++ AMP backend Implementation for Torch7 ** #

##Introduction: ##

This repository hosts the C++ AMP backend implementation project for  [torch7](http://torch.ch/). Torch7 framework currently has a CUDA backend support in the form of [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn) packages. The goal of this project is to develop  gputorch and gpunn packages that would functionally behave as  C++ AMP counterparts for exisiting cutorch and cunn packages. This project mainly targets the linux platform and makes use of the linux-based C++ AMP compiler implementation hosted [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/overview)



##Repository Structure: ##

##Prerequisites: ##
* **dGPU**:  AMD firepro S9150
* **OS** : Ubuntu 14.04 LTS
* **Ubuntu Pack**: libc6-dev-i386
* **AMD APP SDK** : Ver 2.9.1 launched on 18/8/2014 from [here](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* **clBLAS**: ver 2-2.0 from [here](https://github.com/clMathLibraries/clBLAS/releases)
* **AMD Driver installer**: amd-driver-installer-14.301.1001-x86.x86_64


##Building and set up:    
######Need to be a super user

(i) **Torch7 install**:

      *  curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
  
      *  curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-luajit+torch | bash
      


(ii)  ** C++ AMP Compiler installation**: Indepth details can be found [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/overview)

Prepare a directory for work space.

   * mkdir mcw_cppamp

   * cd mcw_cppamp 
   
   * git clone https://bitbucket.org/multicoreware/cppamp-driver-ng.git src

   * git checkout gmac-exp (gmac-exp branch is tailor made for torch7 use case)

Create a build directory and configure using CMake.

  *  mkdir mcw_cppamp/gmac_exp_build

  * cd mcw_cppamp/gmac_exp_build

   * cmake ../src -DCMAKE_BUILD_TYPE=Release (The gmac-exp branch expects the AMDAPP SDK in the path /opt/AMDAPP)

Build the whole system. This will build clang and other libraries that require one time build.

  * make [-j #] world           (# is the number of parallel builds)

  * make                        (this builds llvm utilities)
 
(iii) ** Bolt Set up:**

To set up bolt use the master branch of https://bitbucket.org/multicoreware/cppamp-driver-ng.git

  * cd mcw_cppamp/src

  * git checkout master

 *  mkdir mcw_cppamp/master-build && cd mcw_cppamp/master-build

 * cmake ../src (default options can be overridden on the command line for master branch:
  cmake ../src \
      -DCLANG_URL=https://bitbucket.org/multicoreware/cppamp-ng.git \
      -DOPENCL_HEADER_DIR=/opt/AMDAPP/include \
      -DOPENCL_LIBRARY_DIR=/opt/AMDAPP/lib/x86_64 \ )

*  make [-j #] world

*  make (This builts bolt and other support libraries)

*  copy the Bolt directory from mcw_cppamp/src to mcw_cppamp 

*  cd mcw_cppamp/src 

* git checkout gmac-exp (Need to get back to this)





(iv) **CLBLAS setup**:

Extract the appropriate clBlas binary package from [here](https://github.com/clMathLibraries/clBLAS/releases)

(V) **Building gputorch and gpunn:**

Prior to building these packages the following environment variables need to be set using export command

* AMDAPPSDKROOT=<path to AMD APP SDK>
* CLBLASROOT=<path to clBLAS binary pack>
* BOLTROOT =<path to Bolt Source> (copied from master branch)
* MCWCPPAMPROOT=<path to mcw_cppamp dir>
* LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CLBLASROOT/lib64

After this the following are the steps to build gputorch and gpunn

Build Torch7 Core:

  *  cd mcw_torch7_core

  *  luarocks make rocks/torch-scm-1.rockspec

Build NN pack

  * cd mcw_nn

  * luarocks make rocks/nn-scm-1.rockspec

Build gputorch pack

  * cd gputorch

  * luarocks make rocks/gputorch-scm-1.rockspec

Build gpunn pack

  * cd gpunn

  * luarocks make rocks/gpunn-scm-1.rockspec


Testing gputorch:

* cd gputorch

* th -lgputorch -e "gputorch.test()"


Testing gpunn:

* cd gpunn

* th -lgpunn -e "nn.testgpu()"

Running FaceBook Benchmark:

* tar -xvf imagenet-barebones.tar.gz

* cd imagenet-barebones

* th -i runMetaData.lua

* exit Torch prompt

* th -lgpunn -i main.lua