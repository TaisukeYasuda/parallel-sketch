# Sketches

This directory contains the main implementations of sketching algorithms.

## Interface
All sketches will have the following interface.
```
Sketch {
    void sketch(matrix *A, matrix *SA);
    static size_t eps_approx_rows(double eps, size_t n, size_t d);
}
```
The function `sketch` takes pointers to the original `n x d` matrix `A` and a
pointer to the allocated space for the sketched `p x d` matrix `SA`. The
function `eps_approx_rows` takes the dimensions of the original matrix `A` and
returns the number of rows for the sketch required to guarantee a `1+eps`
subspace embedding guarantee with probability at least 99/100. This success
probability is somewhat arbitrary, and may be boosted by repeating the procedure.

## Description of Sketching Algorithms
* [Gaussian Sketch](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe1.pdf)
* [Count Sketch](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe4.pdf)
* [Leverage Score
  Sampling](http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall17/scribe8.pdf)
