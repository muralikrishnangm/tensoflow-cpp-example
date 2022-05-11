# Simple makefile
# First do `source build/<machine_name>.env`
# Make sure TF_INCLUDE_DIR and TF_LIBRARY_DIR are defined

CC = gcc
CCFLAGS =  -I$(TF_INCLUDE_DIR) -L$(TF_LIBRARY_DIR) -ltensorflow


all: run

run: example_tensorflowNN_cpu.o
	$(CC) $(CCFLAGS) -o example.exe example_tensorflowNN_cpu.cpp
     
clean:
	 rm *.o *.exe 


