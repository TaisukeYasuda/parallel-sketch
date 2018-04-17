template <typename T>
class Sketch {
    public:
        virtual void sketch(matrix *A, matrix *SA, size_t n, size_t d,
            size_t p) = 0;

};
