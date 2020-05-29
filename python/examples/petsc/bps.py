# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

# @file
# CEED BPs example using petsc4py with DMPlex

#!/usr/bin/env python3

import sys
import petsc4py
from petsc4py import PETSc
import libceed
import numpy as np
import math


# -----------------------------------------------------------------------------
# BPs Utilities
# -----------------------------------------------------------------------------


def BCsMass(ncompu, x):
    u = np.zeros(ncompu)
    for i in range(ncompu):
        u[i] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
    return u


def SetupMassRhs():
    # TODO
    return  # TODO


def Error():
    # TODO
    return  # TODO


def BCsDiff(dim, ncompu, x):
    u = np.zeros(ncompu)
    c = np.arange(0, 3, dtype="float64")
    k = np.arange(1, 4, dtype="float64")
    for i in range(ncompu):
        u[i] = math.sin(math.pi * (c[0] + k[0]*x[0])) * \
               math.sin(math.pi * (c[1] + k[1]*x[1])) * \
               math.sin(math.pi * (c[2] + k[2]*x[2]))
        return u


def SetupDiffRhs():
    # TODO
    return  # TODO


def createBCLabel(dm, name):
    dm.createLabel(name)
    label = dm.getLabel(name)
    # Need to mark boundary faces but at the moment there is no
    # interface in petsc4py for DMPlexMarkBoundaryFaces() and
    # DMPlexLabelComplete()
    return dm


# -----------------------------------------------------------------------------
# BP Options
# -----------------------------------------------------------------------------


bp_options = {
    'bp1': {
        'ncompu': 1,
        'qdatasize': 1,
        'qextra': 1,
        'setupgeo': libceed.QFunctionByName("Mass3DBuild"),
        'setuprhs': SetupMassRhs,
        'apply': libceed.QFunctionByName("Mass3DApply"),
        'error': Error,
        'inmode': libceed.EVAL_INTERP,
        'outmode': libceed.EVAL_INTERP,
        'qmode': libceed.GAUSS,
        'enforce_bc': False,
        'bcs_func': BCsMass
    },
    'bp3': {
        'ncompu': 1,
        'qdatasize': 6,
        'qextra': 1,
        'setupgeo': libceed.QFunctionByName("Diff3DBuild"),
        'setuprhs': SetupDiffRhs,
        'apply': libceed.QFunctionByName("Diff3DApply"),
        'error': Error,
        'inmode': libceed.EVAL_GRAD,
        'outmode': libceed.EVAL_GRAD,
        'qmode': libceed.GAUSS,
        'enforce_bc': True,
        'bcs_func': BCsDiff
    }
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def split3(size, m, reverse):
    sizeleft = size
    for d in range(3):
        tmp = int(math.ceil(sizeleft**(1./(3 - d))))
        while tmp * (sizeleft / tmp) != sizeleft:
            tmp += 1
        m[2-d if reverse else d] = tmp
        sizeleft /= tmp

    return m

# -----------------------------------------------------------------------------
# PETSc Setup
# -----------------------------------------------------------------------------


def setupDMByDegree(comm, dm, dim, degree, ncompu, bp_choice):
    # Setup FE
    fe = PETSc.FE().createDefault(dim, ncompu, isSimplex=False, comm=comm)
    dm.setFromOptions()
    dm.addField(fe)

    # Setup DM
    dm.createDS()
    if bp_options[bp_choice]['enforce_bc']:
        if dm.hasLabel('marker'):
            dm = createBCLabel(dm, 'marker')
        # At this point we would use DMAddBoundary() and
        # DMPlexSetClosurePermutationTensor() but there is no interface
        # in petsc4py at the moment

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    dim = 3
    ncompu = 1

    petsc4py.init(sys.argv)
    comm = PETSc.COMM_WORLD
    commsize = comm.getSize()
    rank = comm.getRank()

    OptDB = PETSc.Options()
    PETSc.Cuda

    problem = OptDB.getString('problem', 'bp1')
    benchmark_mode = OptDB.getBool('benchmark', False)
    degree = OptDB.getInt('degree', 3)
    degree = OptDB.getInt('qextra', 1)
    ceed_resource = OptDB.getString('ceed', '/cpu/self')
    cells_x = OptDB.getInt('cells_x', 3)
    cells_y = OptDB.getInt('cells_y', 3)
    cells_z = OptDB.getInt('cells_z', 3)
    requested_memtype_name = OptDB.getString('memtype', 'host')
    local_nodes = OptDB.getInt('local_nodes', 1000)

    melem = [cells_x, cells_y, cells_z]

    # Setup DM
    if local_nodes != 1000:
        # Find a nicely composite number of elements no less than global nodes
        gelem = max([1, commsize * local_nodes / (degree**dim)])
        while True:
            melem = split3(commsize, melem, True)
            if max(melem) / min(melem) <= 2:
                break
        dm = PETSc.DMPlex().createBoxMesh(melem, simplex=False, comm=comm)

    if commsize > 1:
        dm = dm.distribute()

    # Set up libCEED
    ceed = libceed.Ceed(ceed_resource)
    memtype = ceed.get_preferred_memtype()

    # Create DM
    setupDMByDegree(comm, dm, dim, degree, ncompu, problem)
