#include <cstring>

#include "util/transform.h"

void create_trans_from_tu(float *trans, const float *t, const float *u) {
    memcpy(&trans[0], &u[0], 3*sizeof(float));
    memcpy(&trans[4], &u[3], 3*sizeof(float));
    memcpy(&trans[8], &u[6], 3*sizeof(float));
    memcpy(&trans[12], t, 3*sizeof(float));
    trans[15] = 1;
    trans[3] = 0;
    trans[7] = 0;
    trans[11] = 0;
}
