/*
 * Standalone smoke test for the SWI engine. Builds without Python.
 *   make smoke && ./swi_smoke
 * Confirms the shared-library entry point runs and the gap sentinel fires.
 * Channel order: 19V 19H 22V 37V 37H 85V 85H, packed = Kelvin - 70.
 */
#include <stdio.h>
#include "swi_api.h"

int main(void)
{
    /* Three cells, packed (Kelvin - 70):
     *  0: orbital gap / fill (19V well below 100 -> SNOW=-100)
     *  1: warm vegetated-ish land surface
     *  2: cold scattering surface (likely snow/ice routing)
     */
    int chan[3 * 7] = {
        32,  32,  32,  32,  32,  32,  32,    /* fill byte 32 everywhere */
        200, 185, 202, 198, 185, 195, 188,   /* ~270/255/272/268/255/265/258 K */
        150, 148, 160, 175, 170, 185, 178    /* cold, strong scattering */
    };
    float temp[3], wet[3];
    int   snow[3], ret[3];
    int   i;

    swi_eval_packed(chan, 3, temp, wet, snow, ret);

    printf("cell  RTEMP(K)   WET     SNOW  ret\n");
    for (i = 0; i < 3; ++i)
        printf(" %d   %8.3f  %7.2f  %4d  %3d\n",
               i, temp[i], wet[i], snow[i], ret[i]);

    if (snow[0] == -100 && ret[0] == -1) {
        printf("OK: gap sentinel fired for fill cell\n");
        return 0;
    }
    printf("FAIL: gap cell did not return the orbital-gap sentinel\n");
    return 1;
}
