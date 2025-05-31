CC=gcc
AR=ar

CWD=.
SRC_DIR=${CWD}/src
LIB_NAME=scalar_softmax_f32
LIB_SRC_DIR=${SRC_DIR}/lib
BIN_DIR=${CWD}/bin
LIB_DIR=${BIN_DIR}/lib
COMMON_FLAG=-O3 -Wall -Werror
LIB_CFLAG=${COMMON_FLAG} -fno-strict-aliasing -c
APP_CFLAG=${COMMON_FLAG} -Wno-unused-but-set-variable
APP_LFLAG=-l${LIB_NAME} -L${LIB_DIR} -lm

app : lib
	mkdir -p ${BIN_DIR}
	${CC} ${SRC_DIR}/main.c -o ${BIN_DIR}/test ${APP_CFLAG} ${APP_LFLAG}

lib :
	mkdir -p ${BIN_DIR}
	mkdir -p ${LIB_DIR}
	${CC} ${LIB_SRC_DIR}/${LIB_NAME}.c -o ${BIN_DIR}/${LIB_NAME}.o ${LIB_CFLAG}
	${AR} crD ${LIB_DIR}/lib${LIB_NAME}.a ${BIN_DIR}/${LIB_NAME}.o

run : app
	${BIN_DIR}/test -f ${BIN_DIR}/data -g ${BIN_DIR}/golden

clean :
	rm -rf ${BIN_DIR}

RISCV_CC=${RISCV_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc
RISCV_OBJDUMP=${RISCV_TOOLCHAIN}/bin/riscv64-unknown-elf-objdump
RISCV_AR=${RISCV_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc-ar
RISCV_LIB_DIR=${RISCV_TOOLCHAIN}/lib
RISCV_INCLUDE_DIR=${RISCV_TOOLCHAIN}/lib/gcc/riscv64-unknown-elf/13.2.0
RISCV_SPIKE=${SPIKE_PATH}/spike
RVV_LIB_NAME=rvv_softmax_f32
RVV_LIB_CFLAG=${COMMON_FLAG} -fno-strict-aliasing -c
RVV_APP_LFLAG=-l${RVV_LIB_NAME} -L${LIB_DIR} -L${RISCV_LIB_DIR} -lm -I${RISCV_INCLUDE_DIR}
RVV_APP_CFLAG=${COMMON_FLAG} -Wno-unused-but-set-variable

app-rvv : lib-rvv
	mkdir -p ${BIN_DIR}
	${RISCV_CC} ${SRC_DIR}/main.c -DUSE_RVV -o ${BIN_DIR}/rvv-test ${RVV_APP_CFLAG} ${RVV_APP_LFLAG}
	${RISCV_OBJDUMP} ./bin/rvv-test -D > ./bin/dump
	${RISCV_OBJDUMP} ./bin/rvv-test -h > ./bin/mem

lib-rvv :
	mkdir -p ${BIN_DIR}
	mkdir -p ${LIB_DIR}
	${RISCV_CC} ${LIB_SRC_DIR}/${RVV_LIB_NAME}.c -o ${BIN_DIR}/${RVV_LIB_NAME}.o ${RVV_LIB_CFLAG}
	${RISCV_AR} crD ${LIB_DIR}/lib${RVV_LIB_NAME}.a ${BIN_DIR}/${RVV_LIB_NAME}.o

run-rvv : app-rvv
	${RISCV_SPIKE} --isa=rv64gcv ${PK_PATH}/pk ./bin/rvv-test -f ${BIN_DIR}/data -g ${BIN_DIR}/golden

app-riscv : lib-riscv
	mkdir -p ${BIN_DIR}
	${RISCV_CC} ${SRC_DIR}/main.c -o ${BIN_DIR}/riscv-test ${APP_CFLAG} ${APP_LFLAG}
	${RISCV_OBJDUMP} ./bin/riscv-test -D > ./bin/dump
	${RISCV_OBJDUMP} ./bin/riscv-test -h > ./bin/mem

lib-riscv :
	mkdir -p ${BIN_DIR}
	mkdir -p ${LIB_DIR}
	${RISCV_CC} ${LIB_SRC_DIR}/${LIB_NAME}.c -o ${BIN_DIR}/${LIB_NAME}.o ${RVV_LIB_CFLAG}
	${RISCV_AR} crD ${LIB_DIR}/lib${LIB_NAME}.a ${BIN_DIR}/${LIB_NAME}.o

run-riscv : app-riscv
	${RISCV_SPIKE} --isa=rv64gcv ${PK_PATH}/pk ./bin/riscv-test -f ${BIN_DIR}/data -g ${BIN_DIR}/golden