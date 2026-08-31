CC ?= gcc
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

SRCDIR = src
INCDIR = include
BUILDDIR = build
BINDIR = bin

LIB_SRCS = $(SRCDIR)/rv_hdc_core.c \
           $(SRCDIR)/rv_hdc_fpe_a.c \
           $(SRCDIR)/rv_hdc_fpe_b.c \
           $(SRCDIR)/rv_hdc_fpe_c.c \
           $(SRCDIR)/rv_hdc_sequence.c

LIB_OBJS = $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(LIB_SRCS))

DEMO_SRC = examples/demo.c
DEMO_BIN = $(BINDIR)/rv_hdc_demo

.PHONY: all clean run

all: $(DEMO_BIN)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(DEMO_BIN): $(DEMO_SRC) $(LIB_OBJS) | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run: $(DEMO_BIN)
	./$(DEMO_BIN)

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
