#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.PHONY: all clean ir-emit

EMIT_LLVM_IR ?= 0
NCCL_EMIT_LTO_IR ?= 0

# Set up one make target to avoid race
IR_GOALS :=
ifneq ($(EMIT_LLVM_IR), 0)
IR_GOALS += llvm_ir
endif
ifneq ($(NCCL_EMIT_LTO_IR), 0)
IR_GOALS += ltoir
endif

#默认target为编译
default: src.build
ifneq ($(IR_GOALS),)
default: ir-emit
endif

install: src.install
BUILDDIR ?= $(abspath ./build)
#取builddir的绝对目录
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src pkg nccl4py ir
clean: ${TARGETS:%=%.clean}
examples.build: src.build
ir.build: src.build
ir.llvm_ir: src.build
ir.ltoir: src.build
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
	${MAKE} -C docs/examples NCCL_HOME=${ABSBUILDDIR}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

nccl4py.%:
	${MAKE} -C bindings/nccl4py $* BUILDDIR=${ABSBUILDDIR}

# IR generation requires src.build first
ir.%:
	${MAKE} -C bindings/ir $* BUILDDIR=${ABSBUILDDIR}

ir-emit: src.build
	${MAKE} -C bindings/ir $(IR_GOALS) BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
pkg.txz.prep: lic
