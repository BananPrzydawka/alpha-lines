# ARCH defaults to the local GPU's compute capability; falls back to sm_90a
# when no GPU is present. Override explicitly for cross-compilation:
#   make inspect ARCH=sm_90a      (works with no GPU at all)
DETECT := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                  2>/dev/null | head -1 | tr -d '.')
ARCH   ?= sm_$(or $(DETECT),90a)

NVCC   ?= nvcc
SRCDIR := src
BLD    := build/$(ARCH)

SRCS  := $(wildcard $(SRCDIR)/*.cu)
NAMES := $(notdir $(basename $(SRCS)))
BINS  := $(addprefix $(BLD)/,$(NAMES))
PTX   := $(addsuffix .ptx,$(BINS))
SASS  := $(addsuffix .sass,$(BINS))

FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -diag-suppress 177

# --- needs matching hardware to run (compiles anywhere) ---------------------
bins: $(BINS)

$(BLD)/%: $(SRCDIR)/%.cu | $(BLD)
	$(NVCC) $(FLAGS) -o $@ $<

# --- no GPU required --------------------------------------------------------
inspect: $(PTX) $(SASS) regs

$(BLD)/%.ptx: $(SRCDIR)/%.cu | $(BLD)
	$(NVCC) $(FLAGS) -ptx -o $@ $<

$(BLD)/%.cubin: $(SRCDIR)/%.cu | $(BLD)
	$(NVCC) $(FLAGS) -cubin -o $@ $<

$(BLD)/%.sass: $(BLD)/%.cubin
	nvdisasm -c -g $< > $@

regs: | $(BLD)
	@for f in $(SRCS); do echo "== $$f"; \
	  $(NVCC) $(FLAGS) -c --ptxas-options=-v $$f -o /dev/null; \
	done 2>&1 | tee $(BLD)/ptxas.txt

# disassemble one kernel by substring: make dis F=add K=add_vec8
F ?= add
dis: $(BLD)/$(F).cubin
	@cuobjdump -sass -fun $$(cuobjdump -symbols $(BLD)/$(F).cubin \
	   | awk '/STT_FUNC/ && /$(K)/ {print $$NF; exit}') $(BLD)/$(F).cubin

syms: $(BLD)/$(F).cubin
	@cuobjdump -symbols $< | grep STT_FUNC

$(BLD):
	mkdir -p $@

printbld:
	@echo $(BLD)

printarch:
	@echo $(ARCH)

clean:
	rm -rf build out

.PHONY: bins inspect regs dis syms printbld printarch clean
.PRECIOUS: $(BLD)/%.cubin