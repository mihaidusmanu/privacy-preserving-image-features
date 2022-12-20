#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "lib/types.h"
#include "lib/point_to_subspace_matching.cc"
#include "lib/subspace_to_subspace_matching.cc"

PYBIND11_MODULE(pyppifcuda, m) {
    m.doc() = "Privacy-Preserving Image Features CUDA bindings";

    // Point-to-subspace matching.
    m.def(
        "point_to_subspace_exhaustive_matcher", &point_to_subspace_exhaustive_matcher,
        py::arg("points"), py::arg("subspaces")
    );

    // Subspace-to-subspace matching.
    m.def(
        "subspace_to_subspace_exhaustive_matcher", &subspace_to_subspace_exhaustive_matcher,
        py::arg("subspaces1"), py::arg("subspaces2")
    );
}
