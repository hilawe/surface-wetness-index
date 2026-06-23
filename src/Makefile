# Build the Surface Wetness Index engine.
#
#   make            -> shared library libswi.{dylib,so} (the oracle)
#   make smoke      -> standalone C smoke test (swi_smoke)
#   make clean
#
# The shared library wraps the original Basist decision tree (sig_recog.c)
# via swi_api.c so Python can call it through ctypes.

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  LIBEXT  := dylib
  SHFLAGS := -dynamiclib
else
  LIBEXT  := so
  SHFLAGS := -shared
endif

CC     := cc
CFLAGS := -O2 -fPIC -Wall
LIB    := libswi.$(LIBEXT)

.PHONY: all smoke clean
all: $(LIB)

$(LIB): sig_recog.c swi_api.c swi_api.h
	$(CC) $(CFLAGS) $(SHFLAGS) -o $(LIB) sig_recog.c swi_api.c -lm

smoke: swi_smoke.c sig_recog.c swi_api.c swi_api.h
	$(CC) $(CFLAGS) -o swi_smoke swi_smoke.c sig_recog.c swi_api.c -lm

clean:
	rm -f $(LIB) swi_smoke *.o
