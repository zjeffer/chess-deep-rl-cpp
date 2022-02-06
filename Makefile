#OBJS specifies which files to compile as part of the project
OBJS = src/main.cc src/board.cc src/mcts.cc src/move.cc src/node.cc src/chess/thc.cc src/neuralnet.cc src/mapper.cc

#CC specifies which compiler we're using
CC = g++

#COMPILER_FLAGS specifies the additional compilation options we're using
# -w suppresses all warnings
COMPILER_FLAGS = -g -Wall

#LINKER_FLAGS specifies the libraries we're linking against
LINKER_FLAGS = -I /usr/include/tensorflow -ltensorflow

#OBJ_NAME specifies the name of our exectuable
OBJ_NAME = chess-rl

#This is the target that compiles our executable
all : $(OBJS)
	$(CC) $(OBJS) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $(OBJ_NAME)
