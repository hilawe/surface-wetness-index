#ifndef SWI_API_H
#define SWI_API_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Batch wrapper around the Basist signal-recognition decision tree
 * (sig_recog in sig_recog.c). Added for the revival: lets Python (or any
 * caller) evaluate many grid cells through the original C engine, which serves
 * as the exact reference oracle for the vectorized NumPy port.
 *
 * chan : pointer to n*7 ints, row-major, one cell per row. Channel order:
 *          [0]=19V [1]=19H [2]=22V [3]=37V [4]=37H [5]=85V [6]=85H
 *        Values are in the NATIVE PACKED domain expected by sig_recog:
 *          packed = brightness_temperature_Kelvin - 70
 *        i.e. the stored daily 1/3-degree grid byte (0..255). sig_recog adds
 *        the +70 K offset internally.
 * n    : number of cells.
 * temp : out, n floats, land skin temperature RTEMP in Kelvin (-99 undefined).
 * wet  : out, n floats, surface wetness index WET (0..~100; -99 unusable).
 * snow : out, n ints, SNOW flag/scattering magnitude (0,-1,-99,-100,>0).
 * ret  : out, n ints, sig_recog return code (0 good, 1 water-condition, -1 rej).
 *
 * Any of temp/wet/snow/ret may be NULL to skip that output.
 *
 * NOTE: sig_recog uses file-scope globals, so this routine is NOT thread-safe.
 * Call it from a single thread, or guard with a lock.
 */
void swi_eval_packed(const int *chan, int n,
                     float *temp, float *wet, int *snow, int *ret);

#ifdef __cplusplus
}
#endif

#endif /* SWI_API_H */
