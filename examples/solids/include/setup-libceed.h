#ifndef setuplibceed_h
#define setuplibceed_h

#include <ceed.h>
#include <petsc.h>
#include "../include/structs.h"

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data);

// creates subdoman operators for mulitmaterials
PetscErrorCode CreateMaterialSubdomainOpsFromOptions(DM dm, char *prefix, char *cell_set_name, PetscInt *cell_set_value, MaterialSubdomainOps ops);

// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i);

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domain_label, CeedInt *values,
    CeedElemRestriction *elem_restr);

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, CeedInt value, CeedInt P,
                                       CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i);

// Set up libCEED for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dm_energy, DM dm_diagnostic,
                                     Ceed ceed, AppCtx app_ctx,
                                     CeedQFunctionContext phys_ctx,
                                     ProblemData problem_data,
                                     PetscInt fine_level, PetscInt num_comp_u,
                                     PetscInt U_g_size, PetscInt U_loc_size,
                                     CeedVector force_ceed,
                                     CeedVector neumann_ceed, CeedData *data);

// Set up libCEED multigrid level for a given degree
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx app_ctx,
                                 ProblemData problem_data, PetscInt level,
                                 PetscInt num_comp_u, PetscInt U_g_size,
                                 PetscInt U_loc_size, CeedVector fine_mult,
                                 CeedData *data);

#endif // setuplibceed_h
