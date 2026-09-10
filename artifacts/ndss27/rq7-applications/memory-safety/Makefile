CFLAGS = -nostdlib -O0 -fno-stack-protector -no-pie
TARGETS = bof.elf uaf.elf sc.elf aiw.elf

all: $(TARGETS) input.bin

%.elf: %.c syscalls.h
	gcc $(CFLAGS) -o $@ $<

input.bin:
	printf 'XAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' > $@

clean:
	rm -f $(TARGETS) input.bin
