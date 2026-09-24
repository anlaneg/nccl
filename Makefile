#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY: all clean

#默认target为编译
default: src.build
install: src.install
BUILDDIR ?= $(abspath ./build)
#取builddir的绝对目录
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src pkg
clean: ${TARGETS:%=%.clean}
examples.build: src.build
LICENSE_FILES := LICENSE.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	install -m 644 $< $@

#进入到src目录编译（target以$*指出）
src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

#进入examples目录编译
examples: src.build
	${MAKE} -C examples NCCL_HOME=${ABSBUILDDIR}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
pkg.txz.prep: lic
