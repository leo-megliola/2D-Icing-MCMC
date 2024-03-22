#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include "pcg_random.hpp"

namespace py = pybind11;

void test(py::array_t<int>& lattice) {

    py::print("\nHello from C++\n");

    auto L = lattice.mutable_unchecked<2>();

    py::print("Array shape:", L.shape(0), "x", L.shape(1),"\n");

    int rows = L.shape(0);
    int cols = L.shape(1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            L(i,j) = 42;
        }
    }
}

void random_test(py::array_t<int>& lattice) {

    py::print("\nHello from random C++\n");
    auto L = lattice.mutable_unchecked<2>();

    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});
    std::uniform_int_distribution<int> i_dist(0, L.shape(0));

    py::print("Generating random values for array shape:", L.shape(0), "x", L.shape(1),"\n");

    int rows = L.shape(0);
    int cols = L.shape(1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            L(i,j) = i_dist(rng);
        }
    }
}

PYBIND11_MODULE(test_module, m) {
    m.doc() = "Module to test read/write access to numpy array";
    m.def("test", 
          &test,
          "writes to a 2D numpy array",
          py::arg("lattice").noconvert() );
    m.def("random_test", 
          &random_test,
          "writes random values to a 2D numpy array",
          py::arg("lattice").noconvert() );
}