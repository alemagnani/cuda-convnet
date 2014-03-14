SHELL = sh -xv

MODELNAME := _ConvNet

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include -I./include/common -I./include/cudaconv2 -I./include/nvmatrix
LIB := -L/usr/lib/nvidia-current/ -lpthread -L$(ATLAS_LIB_PATH) -L$(CUDA_INSTALL_PATH)/lib64 -lcblas  -lnvToolsExt

USECUBLAS   := 1
USECUSPARSE := 1

PYTHON_VERSION=2.7
LIB += -lpython$(PYTHON_VERSION)

GENCODE_ARCH :=  -gencode=arch=compute_30,code=\"sm_30,compute_30\"

COMMONFLAGS := -DUSE_NVTX -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu src/cudaconv2/*.cu src/nvmatrix/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh include/cudaconv2/*.cuh include/nvmatrix/*.cuh)
CCFILES				:= $(shell echo src/common/*.cpp)
C_DEPS				:= $(shell echo include/common/*.h)

include common-gcc-cuda-5.5.mk

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/cudaconv2
	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/common
	$(VERBOSE)mkdir -p $(TARGETDIR)
