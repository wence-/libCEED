CEED_DIR ?= $(abspath ..)
CEED_FLAGS ?= -I$(CEED_DIR)/include
CEED_LIBS ?= -Wl,-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

# Use an MFEM build directory or an MFEM install directory
MFEM_DIR ?= $(abspath ../../mfem)
CONFIG_MK ?= $(or $(wildcard $(MFEM_DIR)/config/config.mk),$(wildcard\
  $(MFEM_DIR)/share/mfem/config.mk),invalid_mfem_dir)

NEK5K_DIR ?= $(abspath ../../Nek5000)
MPI ?= 1

PETSC_DIR ?=
PETSC_ARCH ?=

.PHONY: all ceed mfem nek petsc fluids solids clean

all: ceed mfem nek petsc fluids solids

ceed:
	make CEED_DIR=$(CEED_DIR) -C ceed all

mfem:
	make CEED_DIR=$(CEED_DIR) MFEM_DIR=$(MFEM_DIR) -C mfem all

# Note: Multiple Nek files cannot be built in parallel. The '+' here enables
#       this single Nek bps file to be built in parallel with other examples,
#       such as when calling `make prove-all -j2` for the top level Makefile.
nek:
	+make CEED_DIR=$(CEED_DIR) NEK5K_DIR=$(NEK5K_DIR) MPI=$(MPI) -C nek bps

petsc:
	make CEED_DIR=$(CEED_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C petsc all

fluids:
	make CEED_DIR=$(CEED_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C fluids all

solids:
	make CEED_DIR=$(CEED_DIR) PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) -C solids all
clean:
	+make -C ceed clean
	+make -C mfem clean
	+make -C nek clean
	+make -C petsc clean
	+make -C fluids clean
	+make -C solids clean
