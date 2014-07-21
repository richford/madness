include $(top_srcdir)/config/MakeGlobal.am

#AM_CPPFLAGS += -DMRA_DATA_DIR="\"`pwd`\""
AM_CPPFLAGS += -DMRA_DATA_DIR=\"$(abs_srcdir)\"

#bin_PROGRAMS = testsuite  testperiodic testbsh testbc testproj mraplot testpdiff testdiff1Db testdiff1D testdiff2D testdiff3D testgconv testopdir testqm testper test6
lib_LIBRARIES = libMADmra.a



#TESTS = testsuite  testperiodic testbc testbsh testproj testpdiff \
#        testdiff1Db testdiff1D testdiff2D testdiff3D testgconv testopdir \
#        testqm testunaryop testper

#TESTS = testbsh testproj testpdiff testper \
#        testdiff1Db testdiff1D testdiff2D testdiff3D testgconv testopdir \
#        testsuite 


data_DATA = autocorr coeffs gaussleg

thisincludedir = $(includedir)/mra
thisinclude_HEADERS = funcimpl.h  indexit.h  legendre.h  operator.h  vmra.h \
                      funcdefaults.h  key.h mra.h      power.h     twoscale.h mraimpl.h


LDADD = libMADmra.a $(LIBLINALG) $(LIBTENSOR) $(LIBMISC) $(LIBMUPARSER) $(LIBWORLD)

libMADmra_a_SOURCES = \
                      startup.cc legendre.cc twoscale.cc \
                      \
                      funcimpl.h  indexit.h  legendre.h  operator.h  vmra.h \
                      funcdefaults.h  key.h mra.h      power.h     twoscale.h mraimpl.h \
                      funcplot.h function_common_data.h



libraries:	$(lib_LIBRARIES)
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir $@ ; done

install-libraries: install-libLIBRARIES install-thisincludeHEADERS
	for dir in $(SUBDIRS) ; do $(MAKE) -C $$dir $@ ; done

