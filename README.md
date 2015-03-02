# ** C++ AMP backend Implementation for Torch7 ** #

##Introduction: ##

This repository hosts the C++ AMP backend implementation project for  [torch7](http://torch.ch/). Torch7 framework currently has a CUDA backend support in the form of [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn) packages. The goal of this project is to develop  gputorch and gpunn packages that would functionally behave as  C++ AMP counterparts for existing cutorch and cunn packages. This project mainly targets the linux platform and makes use of the linux-based C++ AMP compiler implementation hosted [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/overview)



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

   * git checkout gmac-exp-cache-kernel (gmac-exp-cache-kernel branch is tailor made for torch7 use case)
(note that you can also use git checkout origin/gmac-exp-cache-kernel)

Create a build directory and configure using CMake.

  *  mkdir mcw_cppamp/gmac_exp_build_cache

  * cd mcw_cppamp/gmac_exp_build_cache

   * cmake ../src -DCMAKE_BUILD_TYPE=Release (The gmac-exp-cache-kernel branch expects the AMDAPP SDK in the path /opt/AMDAPP)

Build the whole system. This will build clang and other libraries that require one time build.

  * make [-j #] world           (# is the number of parallel builds. Generally it is # of CPU cores)

  * make                        (this builds llvm utilities)

Note that you might need to manually check updates from C++ AMP Compiler.
Please do the following and rebuild the Compiler if any update is available

```
#!python
 # check updates from C++AMP Compiler
 cd mcw_cppamp/src
 git fetch --all
 git checkout origin/gmac-exp-cache-kernel

 # check updates from C++AMP Compiler's dependency
 cd mcw_cppamp/src/compiler/tools/clang
 git fetch --all
 git checkout origin/master
```

(iii) ** Bolt Set up:**

Bolt binaries are automatically built in (ii)

  * git checkout gmac-exp-cache-kernel (Need to get back to this)

(iv) **CLBLAS setup**:

Extract the appropriate clBlas binary package from [here](https://github.com/clMathLibraries/clBLAS/releases)

(V) **Building gputorch and gpunn:**

Prior to building these packages the following environment variables need to be set using export command

* AMDAPPSDKROOT=<path to AMD APP SDK>
* CLBLASROOT=<path to clBLAS binary pack>
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

If fails, run the following to get detailed messages if any,
 
```
#!python
 th
 require ('gputorch')
```


Testing gpunn:

* cd gpunn

* th -lgpunn -e "nn.testgpu()"

Running FaceBook Benchmark:

Dowload the benchmark from [here](https://multicorewareinc.egnyte.com/dl/31EBfMX0vr)

* tar -xvf imagenet-barebones.tar.gz

* cd imagenet-barebones

* th -i runMetaData.lua

* exit Torch prompt

* th -lgpunn -i main.lua