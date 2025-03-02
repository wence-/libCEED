#ifndef libceed_solids_examples_setup_dm_h
#define libceed_solids_examples_setup_dm_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include "../include/structs.h"

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]);

// Read mesh and distribute DM in parallel
PetscErrorCode CreateDistributedDM(MPI_Comm comm, AppCtx app_ctx, DM *dm);

// Setup DM with FE space of appropriate degree
PetscErrorCode SetupDMByDegree(DM dm, AppCtx app_ctx, PetscInt order,
                               PetscBool boundary, PetscInt num_comp_u);

#endif // libceed_solids_examples_setup_dm_h
