# kaleidoscope
- My take on the kaleidoscope from LLVM (a toy language)

## To Compile the Compiler
```bash
> clang++ -std=c++17 main.cpp -lLLVM -I/opt/homebrew/Cellar/llvm/17.0.6_1/include/ -L/opt/homebrew/Cellar/llvm/17.0.6_1/lib/ -o build/kc
> ./build/kc tests/test2.kal tests/output/test.o
> clang tests/test.c tests/output/test.o -o tests/output/main
> ./tests/output/main
```
