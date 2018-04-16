# Sketches

This directory contains the main implementations of sketching algorithms.

## Interface
All sketches will have the following interface.
```
Sketch {
    void sketch(matrix *A, matrix *SA, size_t n, size_t d, size_t p)
}
```
The sketch function takes pointers to the original `n x d` matrix, a pointer to
the allocated space for the sketched `p x d` matrix, and their dimensions. This
is the main operation, and we will provide different implementations based on
variants of this operation, for instance sparse vs dense matrices, in-place
sketching, etc.

## Description of Sketching Algorithms
* [Gaussian Projection](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe1.pdf)
* [Count Sketch](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe4.pdf)
* [Leverage Score
  Sampling](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe8.pdf)
