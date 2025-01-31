.PHONY: clean

all: asmpt

asmpt.o: asmpt.asm
	nasm -f elf64 -F dwarf -g -o asmpt.o asmpt.asm

asmpt: asmpt.o
	ld -o asmpt asmpt.o

clean:
	rm -f asmpt asmpt.o
