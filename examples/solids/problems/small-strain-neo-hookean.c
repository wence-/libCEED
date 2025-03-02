#include <ceed.h>
#include "../include/structs.h"
#include "../include/setup-libceed.h"
#include "../problems/problems.h"
#include "../problems/neo-hookean.h"
#include "../qfunctions/common.h"
#include "../qfunctions/small-strain-neo-hookean.h"

static const char *const field_names[] = {"gradu"};
static CeedInt field_sizes[] = {9};

ProblemData small_strain_neo_Hookean = {
  .setup_geo = SetupGeo,
  .setup_geo_loc = SetupGeo_loc,
  .q_data_size = 10,
  .quadrature_mode = CEED_GAUSS,
  .residual = ElasSSNHF,
  .residual_loc = ElasSSNHF_loc,
  .number_fields_stored = 1,
  .field_names = field_names,
  .field_sizes = field_sizes,
  .jacobian = ElasSSNHdF,
  .jacobian_loc = ElasSSNHdF_loc,
  .energy = ElasSSNHEnergy,
  .energy_loc = ElasSSNHEnergy_loc,
  .diagnostic = ElasSSNHDiagnostic,
  .diagnostic_loc = ElasSSNHDiagnostic_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasSSNH(DM dm, DM dm_energy,
    DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
    PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector force_ceed, CeedVector neumann_ceed,
    CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx,
                               phys_ctx, small_strain_neo_Hookean,
                               fine_level, num_comp_u, U_g_size, U_loc_size,
                               force_ceed, neumann_ceed, data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasSSNH(DM dm, Ceed ceed, AppCtx app_ctx,
    PetscInt level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
    CeedVector fine_mult, CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedLevel(dm, ceed, app_ctx, small_strain_neo_Hookean,
                           level, num_comp_u, U_g_size, U_loc_size, fine_mult, data);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
