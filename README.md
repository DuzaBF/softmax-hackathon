# softmax-hackathon

This is a solution to the improving softmax challenge on the [RISC-V Online Hackathon](https://community.riscv.org/events/details/risc-v-international-risc-v-academy-presents-risc-v-hackathon-online/).

Generate the test data and golden image with the Jupyter notebook.

## Scalar solution

This solution uses pure C and loops. Can be built and executed on the host machine.

```bash
make run
```

## RVV solution

Cross-compiles to the RISC-V core machine and uses Spike simulator to run the program.
Provide path to RISC-V gcc and Spike with proxy kernel:
```bash
export RISCV_TOOLCHAIN={install_dir}/riscv-gcc
export SPIKE_PATH={install_dir}/riscv-isa-sim/bin
export PK_PATH={install_dir}/riscv64-unknown-elf/bin
```

```bash
make run-rvv
```

## ACE custom instruction

`exp.ace` contains the instruction for the [custom ACE extension](https://www.andestech.com/en/products-solutions/andes-custom-extension/).