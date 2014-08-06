#!/bin/bash

make test_AST > dump
export MAD_NUM_THREADS=4
./test_AST 

