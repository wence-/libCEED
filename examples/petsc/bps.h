#ifndef bps_h
#define bps_h

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// MemType Options
static const char *const mem_types[] = {"host","device", "memType",
                                        "CEED_MEM_", 0
                                       };

// Coarsening options
typedef enum {
  COARSEN_UNIFORM = 0, COARSEN_LOGARITHMIC = 1
} CoarsenType;
static const char *const coarsen_types [] = {"uniform", "logarithmic",
                                             "CoarsenType", "COARSEN", 0
                                            };

static const char *const bp_types[] = {"bp1", "bp2", "bp3", "bp4", "bp5", "bp6",
                                       "BPType", "CEED_BP", 0
                                      };

#endif // bps_h
