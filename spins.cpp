#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include "pcg_random.hpp"

namespace py = pybind11;

void spins(py::array_t<int>& lattice, py::array_t<double>& mean, int N, double B, double T, int J, int KB) {

    auto L = lattice.mutable_unchecked<2>();
    auto M = mean.mutable_unchecked<1>();    //the lattice and mean can now be written into (locally called L and M)

    int rows = L.shape(0);
    int cols = L.shape(1);

    // set up random number generators
    pcg32 rng(pcg_extras::seed_seq_from<std::random_device>{});
    std::uniform_int_distribution<int> ij_dist(0, rows-1);    // [0, rows-1]
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);  // [0.0, 1.0)

    // initialize total magnetization (just do this once, then increment)
    int total = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += L(i,j);
        }
    }

    for (int n = 0; n < N; n++) {
        // pick a random node in the lattice
        int i = ij_dist(rng);
        int j = ij_dist(rng);
        // calculate delta energy (when using index subtraction: add row/col s.t. mod result is positive)     
        int delta_energy  = -J * -2 * L(i,j) * L( (i-1+rows)%rows, j );
        delta_energy     += -J * -2 * L(i,j) * L( (i+1)%rows     , j );
        delta_energy     += -J * -2 * L(i,j) * L( i              ,(j-1+cols)%cols );
        delta_energy     += -J * -2 * L(i,j) * L( i              ,(j+1)%cols );
        delta_energy     += -B * L(i,j);
    
        if (delta_energy <= 0) {
            total -= L(i,j);
            L(i,j) *= -1;
            total += L(i,j);
        } else {
            double prob = std::exp((-1 * (double) delta_energy) / ((double) KB * (double) T)); 
            if (p_dist(rng) < prob) {
                total -= L(i,j);
                L(i,j) *= -1;
                total += L(i,j);
            }               
        }
        M(n) = total;
    }
    // calculate mean magnetization
    double div = (double) (rows * cols);
    for (int n = 0; n < N; n++) {
        M(n) /= div;
    }
}

PYBIND11_MODULE(spins_module, m) {
    m.doc() = "Module to preform MCMC by repeadly updateing spins within the lattice";
    m.def("spins", 
          &spins,
          "writes MCMC results in-place",
          py::arg("lattice").noconvert(), 
          py::arg("mean"),
          py::arg("N"),
          py::arg("B"),
          py::arg("T"),
          py::arg("J"),
          py::arg("KB"));
}