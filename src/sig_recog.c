/* 
  NCDC MONITORING CODE - 
  SSMI signal recognition algorithm

 14 & 19dec2001 Improvements to the snow algorithm including changing the
    snow variable to the scattering signature when NOT ice (:= -1).

  2feb2001 Many changes made on 1/30 & 1/31 are made and noted from the 
   trip to Huntsville for Leslie Litten's work.

  26apr2000 many refinements have been made to this software since the last
     update. The major one occurs on this date when the "prefilter" routine
     is added for removing F08 observations that we do not have enough
     information to adjust between March 1990 and December 1991. Also added
     the regeneration of 85H as well as 85V.

  7may1998 version simplifies the correction coefficients down to the 
     F11DS (am) - the InterSatellite Correction will hopefully deal with
     the drifts and discrepancies between satellites.
   Preceeding version is sig_recog.pre_f11ds.c

  27apr98 version contains estimations for SNOW, and wetness as
    well as temperature.  cw
 Preceeding version sig_recog.c  

  1Nov04 version cuts out unused functions and routines for a  more streamlined version, main signal recognition codes are                                                                                                                          unchanged*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define NV 70

#define RAIN_WET 1
#define WET_SURF 2

float RTEMP, WET, P, PD19;
int SNOW;

void gen_WET_surf(int hour, int P1, int P5, int P6, float F31, float F64)
{
  P = .0783*F31 - .4369*P5 + 1.4828*P6 - .6628*F64;
  /* WET = P - 1.065 * P1;  */
  WET = ((1. - (P1 / (P/1.06))) / .33) * 100.;  
}

int sig_recog(int hour, int P1, int P2, int P3, int P4, int P5, int P6, int P7)
{
  char temp_str[5], pcp_str[5];
  float PD37, F64, F31, F34, PD85, F36, 
    F41, F61, F14, F46, F13, F16, F25, F57, F43;
  int CONTAM;  
  float tt;
  int scat, lime, nop, wets;
  int save_P1;            /* REVIVAL FIX: declaration lost in 2004 streamlining */

  /*  ORBITAL GAP  */     /* REVIVAL FIX: malformed comment from Word extraction */

  /*TEST 1  */
  if(P1 < 100)
  {
    RTEMP = -99.9;
    WET = -99.9;
    SNOW = -100;
    return(-1);
  }

  P1 += NV;
  P2 += NV;
  P3 += NV;
  P4 += NV;
  P5 += NV;
  P6 += NV;
  P7 += NV;
  save_P1 = P1;
  
  /* polarization and scattering */
  PD19 = P1 - P2;
  PD37 = P4 - P5;
  PD85 = P6 - P7;
  F13 = P1 - P3;
  F14 = P1 - P4;
  F16 = P1 - P6;
  F25 = P2 - P5;
  F31 = P3 - P1;
  F34 = P3 - P4;
  F36 = P3 - P6;
  F41 = P4 - P1;
  F43 = P4 - P3;
  F46 = P4 - P6;
  F57 = P5 - P7;
  F61 = P6 - P1;
  F64 = P6 - P4;

  /* VEG LAND (default) */
  RTEMP = 1.0650 * P3;
  WET = 0.0;
  SNOW = 0;
  nop = 1;

/* FILTERS */

  /* TEST 2 */
  if(PD19 < -1 || PD37 < -1 || PD85 < -1 || PD19 > 45)
  {
    RTEMP = -99;
    WET = -99.;
    SNOW = -99;
    return(-1);
  }  

  /* TEST 3 */
  if(F64 > 50 || F64 < -50 || F13 > 30 || F13 < -30 || F34 > 50)
  {
    RTEMP = -99;
    WET = -99.;
    SNOW = -99;
    return(-1);
  }

  /* TEST 4 */
  if(P3<=210 || (P3<=229 && P6<=240 && (F36 < 0 || F46 < 0))) 
  {
    SNOW = -1;
    WET = 0.0;
    RTEMP = -99;
    return(-1);
  }

  if(F31>=1 && PD85>=15 && (P4>P3)) 	/* TEST 5 */
  {
/* added "if" 04Feb02 */
    if((PD19/10 <= F31 || PD19 < 25) && P6 > P1)	/* TEST 6 */
    {
      gen_WET_surf(hour,  P1,  P5,  P6,  F31,  F64);
    }
    else
    {
      WET = -99.;
    }    
    SNOW = 0;
    RTEMP = -99;
    return(-1);
  } 
  
CONTAM = 0;
/* TEST 7 */
    if(PD37 <= 7)
  {
    if(F46>3 && P1>=257 && PD85<=7)  	/* TEST 8 */
    {
      CONTAM = F46;
    }
    else if(F31>0 && F43>0 && F64<=0)	/* TEST 9 */
    {
      CONTAM = F43;
    }  
    if(CONTAM > 25 || (CONTAM > 0 && P1 < 261)) /* TEST 10 */
    {
      WET = 0.0;
      SNOW = 0;
      RTEMP = -99.0;
      return(1);
    }  
  }

  /* SNOW FILTER*/
  scat = F36;
  if(F14>scat) scat = F14;  /* TEST 11 */
  if(F46>scat) scat = F46;  /* TEST 12 */
  SNOW = 0;
  if(scat>=1.0 && P3 < 257)	/* TEST 13 */
  {
    /* turn on SNOW, but look for exceptions */
    SNOW = scat;

    if(PD85 >= 2.5 * (float)SNOW)	/* TEST 14 */
    {
      SNOW = 0;
    } 
    if(P3>=258 || P3>=165.0 + 0.49 * P6 ||
      (P3>=254 && scat<=2.0 && PD85>=3)) /* TEST 15 */
    {
      SNOW = 0;
    }
    if(PD19>=18 && F14<=10 && F46<=5 && F31 <= 0 && P3 > 235) /* TEST 16 */
    {
      if(P3 > 235 && PD37 < 30) /* TEST 17*/
      {
        SNOW = 0;  
      }
      else
      {
        SNOW=0;
        RTEMP = -99.0;
        WET = -99.;
        return(1);
      }  
    }
    if(SNOW != 0 && PD19>=12 && scat<=2 && F14<=2) /* TEST 18*/
    { 
      RTEMP = -99.0;
      WET = -99.;
      SNOW = -99;
      return(1);
    }
    if((F34 > 17 || P6 > P4 && P6 > 245) && (F31 > 10 || PD85 > 20)) 	/* TEST 19*/
    {
      RTEMP = -99.0;
      WET = -99.;
      SNOW = -99;
      return(1);
    }
  }

  /* remove the SNOW that is left */
  if(SNOW > 0) /*TEST 20*/
  {
    RTEMP = -99.0;
    WET = 0.0;
    return(1);
  }       

  /* THIS IS FOR RAIN OR SNOW OVER A WET SURFACE */
  wets = 0;
  if(CONTAM>0 && F36 > 0 && PD37 < 7)      /*TEST 21*/
  {
    if(F31>=-3 && PD85<=10)	/*TEST 22*/
    {
      wets = RAIN_WET;
    
      RTEMP = 1.0714 * P3 + .2183 * F36;
      if(RTEMP < 271.0) 	/*TEST 23*/
      {
        RTEMP = -99.0;
        SNOW = F36;
        WET = 0.0;
        return(1);
      }
      nop = 0;
      WET = -99.;
      return(1);
    }
    else
    {
      RTEMP = -99.0;
      SNOW = 0;
      WET = -99.;
      return(1);
    }
  }

  /* Rain? */
  if((F46>5 && P3>=257 && PD85<=5 && PD37 < 7) || 
     (P3>=257 && F46>10 && PD37 < 7) )        /*TEST 24*/
  {
      RTEMP = -99.0;
      WET = -99.;
      SNOW = -99;
      return(1);
  } 

  /* A WET SURFACE */
  if((F31>3 || F64*2>=PD37 || F41*7>PD19)  && CONTAM == 0 &&
    SNOW == 0 && F64>0)	/*TEST 25*/
  {
    gen_WET_surf(hour,  P1,  P5,  P6,  F31,  F64);
    if(WET>0)	/*TEST 26*/
    {
      if(PD19 > F64 * 4 && F64 >= 5)		/*TEST 27*/
      {
        RTEMP = -99.0;
        WET = -99.;
        SNOW = -99;
        return(1);
      }
      if(F34>0 && F46 > 0)	/*TEST 28*/
      {
        if(PD19 - F46 >= 8)	/*TEST 29*/
        {
          RTEMP = -99.0;
          SNOW = 0;
          WET = 0.0;
          return(1);
        }      
        RTEMP = .3204 * PD19 + 1.0558 * P3 - .5008 * F46;
      }
      else
      {
        RTEMP = P;
      }
      nop = 0;
      wets = WET_SURF;
    }
    else
    {
      WET = 0.;
      RTEMP = .5195 * P1 + 1.0869 * P3 - 0.5375 * P4;
      SNOW = 0;
      return(1);
    }
  }

  /* GLACIAL FILTER*/
/*TEST 30*/
  if(WET != 0.0 && RTEMP <= 258.0 && P6 < 256)
  {
    SNOW = -1;
    WET = -99.;
    RTEMP = -99.;
    return(1);
  } 

  /* Quartz */
  if(P1 >= P3 && PD19 > 25 && P3 + 2 <= P4 && P4 > P6) /*TEST 31*/
  {
    nop = 0;
  }

  /* LIMESTONE */
  lime = 0;
  if( wets == 0 && P4 < P6 && PD37>6)  /*TEST 32*/
  {
    nop = 0;
    lime = 1;
    RTEMP = 0.31091196*PD19 + 0.56659491*P4 + 0.47783562*P6;
    WET = 0.;
    SNOW = 0;
    return(1);
  }

  /* more filters */
  if((PD19 > 20 && PD19+2 < PD37) || (PD19 > 10 && PD19+4 < PD37 && PD85 > PD37)) /*TEST 33*/
  {
    RTEMP = -99;
    WET = -99.;
    SNOW = 0;
    return(1);
  }

  if(P1<=256 && F36<=-4 && WET<=0.0 && PD85>5) 	/*TEST 34*/

  {
    SNOW = -99;
    WET = -99.;  
    RTEMP = -99.;
    return(1);
  }

  if(WET<= 0.0 && PD19 > 8 && (PD85 > PD37 || PD85 > PD19)) /*TEST 35*/
  {
    if((abs(PD19) + abs(PD37) + abs(PD85)) >= 5)	/*TEST 36*/
    {
      RTEMP = -99;
      WET = -99.;
      SNOW = -99;
      return(-1);
    }
  }    

/*TEST 37*/
  if((PD37 > 39 && WET > 0) || (PD85 > 20 && WET == 0 && PD37 < 30) || 
     (wets == WET_SURF && F36 > 0) || (PD19 > 10 && WET == 0 && F64 > 3 
     && lime == 0))
  {
    RTEMP = -99.;
    WET = -99.;
    SNOW = -99;
    return(1);
  }
  if(P3 > P1 && P3 > P6)	/*TEST 38*/
  {
    RTEMP = 1.0730 * P3 + .2260 * F36;
    WET = -99.;
    SNOW = 0;
    nop = 0;
  }
  
  if(WET < 0.) WET = 0.;	/*TEST 39*/
  
  if(RTEMP == -99) return(-1);	/*TEST 40*/

  /* Vegetation is LAST */
  if(nop == 1) 	/*TEST 41*/
  {
    if(F31 > 0 && PD37 > 10) 	/*TEST 42*/
    {
      RTEMP = -99.;
      return(1);
    }
    else
    {
      RTEMP = 1.0698 * P3;
    }  
  }

  return(0);
}

void eps_est_f(int P1, int P2, int P3, int P4, int P5, int P6, int P7,
  int minobs, float *temp, float *wet, int *snw)
{
  /* Fortran friendly routine */
  P1 -= NV;
  P2 -= NV;
  P3 -= NV;
  P4 -= NV;
  P5 -= NV;
  P6 -= NV;
  P7 -= NV;
  
  sig_recog(0, P1, P2, P3, P4, P5, P6, P7);
  *temp = RTEMP;
  *wet = WET;
  *snw = SNOW;

  return;
}
