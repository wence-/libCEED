#ifndef cl_problems_h
#define cl_problems_h

// Problem options
typedef enum {
  ELAS_LINEAR = 0, ELAS_SS_NH = 1, ELAS_FSInitial_NH1 = 2, ELAS_FSInitial_NH2 = 3, ELAS_FSInitial_NH_AD = 4,
  ELAS_FSCurrent_NH1 = 5, ELAS_FSCurrent_NH2 = 6, ELAS_FSInitial_MR1 = 7
} problemType;
static const char *const problemTypes[] = {"Linear",
                                           "SS-NH",
                                           "FSInitial-NH1",
                                           "FSInitial-NH2",
                                           "FSInitial-NH-AD",
                                           "FSCurrent-NH1",
                                           "FSCurrent-NH2",
                                           "FSInitial-MR1",
                                           "problemType","ELAS_",0
                                          };
static const char *const problemTypesForDisp[] = {"Linear elasticity",
                                                  "Hyperelasticity small strain, Neo-Hookean",
                                                  "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
                                                  "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u), C_inv, constant storage",
                                                  "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u), Swork, tape storage, w/ Enzyme AD",
                                                  "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
                                                  "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxcurr, tau, constant storage",
                                                  "Hyperelasticity finite strain Initial configuration Moony-Rivlin w/ dXref_dxinit, Grad(u) storage"
                                                 };

#endif // cl_problems_h