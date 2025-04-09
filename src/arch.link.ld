/* This file is used to generate link.ld, Ara's linker script,
   which depends on the number of lanes of the current configuration */

OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS
{
  . = 0x10000;
  .text : { *(.text) }
  .data : { 
    __global_pointer$ = . + 0x800;
    *(.data)
  }
  .bss : {
    __bss_start = .;
    *(.bss)
    __bss_end = .; 
  }
  .fromhost : {
    fromhost = .;
    . = . + 0x10;
  }
  .tohost : {
    tohost = .;
    . = . + 0x10;
  }
  _end = .;
}