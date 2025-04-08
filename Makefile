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
APP_LFLAG=-l${LIB_NAME} -l${LIB_NAME} -L${LIB_DIR}

app : lib
	mkdir -p ${BIN_DIR}
	${CC} ${SRC_DIR}/main.c -o ${BIN_DIR}/test ${APP_CFLAG} ${APP_LFLAG}

lib :
	mkdir -p ${BIN_DIR}
	mkdir -p ${LIB_DIR}
	${CC} ${LIB_SRC_DIR}/${LIB_NAME}.c -o ${BIN_DIR}/${LIB_NAME}.o ${LIB_CFLAG}
	${AR} crD ${LIB_DIR}/lib${LIB_NAME}.a ${BIN_DIR}/${LIB_NAME}.o

run : app
	${BIN_DIR}/test

clean :
	rm -rf ${BIN_DIR}