# Simple makefile
# First do `source build/<machine_name>.env`
# Make sure TF_INCLUDE_DIR and TF_LIBRARY_DIR are defined

# define which source file to compile. Options: example_tensorflow_HelloWorld , example_tensorflow_NNmodel, example_tensorflow_tanh example_tensorflow_AICT
SRC=example_tensorflow_AICT

all: run

run: $(SRC).o
	$(CC) $(CFLAGS) -o example.exe $(SRC).c
     
clean:
	 rm *.o *.exe 


