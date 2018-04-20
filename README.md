# Parallel Sketching on GPUs

## Organization
```
.
├── docs
├── src                   # source implementation directory
│   ├── sketches          # sketching algorithms
│   ├── applications      # applications of sketching algorithms
├── tests                 # testing directory
│   ├── random_matrices   # randomly generated matrices
│   ├── images            # grayscale images
│   ├── uf_sparse         # UF sparse matrix collection
│   ├── uci_msd           # UCI YearPredictionMSD
│   ├── uci_cover         # UCI Covertype
```

## Building the test
This repository is designed around an out-of-source build using CMake. To build
the project, run the following commands at the root directory of the project:
```bash
parallel-sketch$ mkdir build
parallel-sketch/build$ cmake ..
parallel-sketch/build$ ./sketch_test
```
Note that CMake copies the directory `test/data` into the `build` directory, so
the testing code written in `test` may refer to `./data` and still access these
files.
