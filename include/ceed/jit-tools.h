#ifndef _ceed_jit_h
#define _ceed_jit_h

#include <ceed/ceed.h>

CEED_EXTERN int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer);

#endif
