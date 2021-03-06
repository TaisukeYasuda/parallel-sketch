# Parallel Sketching on GPUs
This project provides sequential and parallel implementations of various matrix
sketching algorithms and downstream approximation algorithms. Our sequential
implementation is written in C++ primarily for Boost matrices, while our
parallel implementation is written in CUDA for NVIDIA GPUs.

## Directory organization
```
.
├── docs
├── include                      # header files
├── src                          # source implementation
│   ├── cpp                      # C++ sequential implementation
│   │   ├── sketches             # sketching algorithms
│   │   └── applications         # downstream applications
│   └── cuda                     # CUDA parallel implementation
│       ├── sketches             # sketching algorithms
│       └── applications         # downstream applications
└── test                         # tests
    └── data                     # testing data
        ├── images               # grayscale images
        ├── random_matrices      # randomly generated matrices
        ├── uci_cover            # UF sparse matrix collection
        ├── uci_msd              # UCI YearPredictionMSD
        └── uf_sparse            # UCI Covertype
```

## Building the test
This repository is designed around an out-of-source build using CMake. First
buid the necessary dependencies by going to the `deps` directory and following
the instructions there. Then, to build the project, run the following commands
at the root directory of the project:
```bash
parallel-sketch$ mkdir build
parallel-sketch/build$ cmake ..
parallel-sketch/build$ make
parallel-sketch/build$ ./count_sketch_test
```
Note that CMake copies the directory `test/data` into the `build` directory, so
the testing code written in `test` may refer to `./data` and still access these
files.
