NEK5K_DIR ?= $(abspath ../../../Nek5000)
MPI ?= 1

ifeq ($(MPI),0)
  ifeq (,$(filter-out undefined default,$(origin CC)))
    CC = gcc
  endif
  ifeq (,$(filter-out undefined default,$(origin FC)))
    FC = gfortran
  endif
else
  FC = mpif77
  CC = mpicc
endif

CEED_DIR ?= $(abspath ../..)
CEED_FLAGS ?= -I$(CEED_DIR)/include
CEED_LIBS ?= -Wl,-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

.PHONY: all bps clean

all: bps

bps: bps/bps.usr
	FC=$(FC) CC=$(CC) MPI=$(MPI) NEK5K_DIR=$(NEK5K_DIR) CEED_DIR=$(CEED_DIR) && ./nek-examples.sh -m

clean:
	NEK5K_DIR=$(NEK5K_DIR) && ./nek-examples.sh -clean
