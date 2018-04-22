# Dependencies
This directory contains dependencies that were not included in our system.

### [Eigen 3.3.4](http://eigen.tuxfamily.org/index.php?title=Main_Page)
In our repository, Eigen is set up as follows.
```bash
parallel-sketch/deps$ mkdir eigen-3.3.4
parallel-sketch/deps$ tar -zxvf eigen-eigen-5a0156e40feb.tar.gz
parallel-sketch/deps$ cd eigen-eigen-5a0156e40feb
parallel-sketch/deps/eigen-eigen-5a0156e40feb$ mkdir build
parallel-sketch/deps/eigen-eigen-5a0156e40feb$ cd build
parallel-sketch/deps/eigen-eigen-5a0156e40feb/build$ cmake ..
parallel-sketch/deps/eigen-eigen-5a0156e40feb/build$ cmake . -DCMAKE_INSTALL_PREFIX=../../eigen-3.3.4
parallel-sketch/deps/eigen-eigen-5a0156e40feb/build$ make install
```
This installs Eigen in the directory `deps/eigen-3.3.4`.
