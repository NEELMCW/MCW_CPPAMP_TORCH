# ** C++ AMP backend Implementation for Torch7 ** #

##Introduction: ##

This repository hosts the C++ AMP backend implementation project for  [torch7](http://torch.ch/). Torch7 framework currently has a CUDA backend support in the form of [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn) packages. The goal of this project is to develop  gputorch and gpunn packages that would functionally behave as  C++ AMP counterparts for exisiting cutorch and cunn packages. This project mainly targets the linux platform and makes use of the linux-based C++ AMP compiler implementation hosted [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/overview)



##Repository Structure: ##

##Prerequisites: ##
* **dGPU**:  AMD firepro S9150
* **OS** : Ubuntu 14.04 LTS
* **Ubuntu Pack**: libc6-dev-i386
* **AMD APP SDK : Ver 2.9.1 launched on 18/8/2014 from [here](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* **clBLAS**: ver 2-2.0 from [here](https://github.com/clMathLibraries/clBLAS/releases)


##Building and set up:    
######Need to be a super user

(i) **Torch7 install**:

      *  curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
  
      *  curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-luajit+torch | bash
      


(ii)  ** C++ AMP Compiler installation**: Indepth details can be found [here](https://bitbucket.org/multicoreware/cppamp-driver-ng/overview)

Prepare a directory for work space.

   * mkdir cppamp

   * cd cppamp 
   
   * git clone https://bitbucket.org/multicoreware/cppamp-driver-ng.git src

(iii)  Bolt Set up:

(iv) CLBLAS setup:

(