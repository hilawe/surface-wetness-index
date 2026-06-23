#include "swi_api.h"

/* Defined in sig_recog.c (file-scope globals and the decision-tree routine). */
extern int   sig_recog(int hour, int P1, int P2, int P3,
                       int P4, int P5, int P6, int P7);
extern float RTEMP, WET;
extern int   SNOW;

void swi_eval_packed(const int *chan, int n,
                     float *temp, float *wet, int *snow, int *ret)
{
    int i;
    for (i = 0; i < n; ++i) {
        const int *c = chan + 7 * i;
        int r = sig_recog(0, c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        if (temp) temp[i] = RTEMP;
        if (wet)  wet[i]  = WET;
        if (snow) snow[i] = SNOW;
        if (ret)  ret[i]  = r;
    }
}
