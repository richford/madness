include $(top_srcdir)/config/MakeGlobal.am

bin_PROGRAMS     = projPsi test hello lineplot wfSlice toDX
projPsi_SOURCES  = projPsi.cc extra.cc extra.h wavef.cc wavef.h hyp.cc hyp.h mpreal.cc 
lineplot_SOURCES = lineplot.cc
test_SOURCES     = test.cc wavef.cc wavef.h hyp.cc hyp.h mpreal.cc interp.h 
wfSlice_SOURCES= wfSlice.cc 
toDX_SOURCES     = toDX.cc wavef.cc wavef.h hyp.cc hyp.h mpreal.cc interp.h 
hello_SOURCES    = hello.cc wavef.cc wavef.h hyp.cc
projPsi_LDADD    = $(MRALIBS)
test_LDADD       = $(MRALIBS)
hello_LDADD      = $(MRALIBS)
lineplot_LDADD   = $(MRALIBS)
wfSlice_LDADD  = $(MRALIBS)
toDX_LDADD  = $(MRALIBS)
