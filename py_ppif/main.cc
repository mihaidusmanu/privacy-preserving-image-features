#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "lib/types.h"
#include "lib/lifting.cc"
#include "lib/point_to_subspace_matching.cc"
#include "lib/subspace_to_subspace_matching.cc"

PYBIND11_MODULE(pyppif, m) {
    m.doc() = "Privacy-Preserving Image Features bindings";

    // Lifting.
    m.def(
        "random_lifting", &random_lifting,
        py::arg("descriptors"), py::arg("subspace_dim"),
        py::arg("seed") = 0
    );
    m.def(
        "adversarial_lifting", &adversarial_lifting,
        py::arg("descriptors"), py::arg("subspace_dim"),
        py::arg("database"), py::arg("num_sub_databases") = 1,
        py::arg("seed") = 0
    );
    m.def(
        "hybrid_lifting", &hybrid_lifting,
        py::arg("descriptors"), py::arg("subspace_dim"),
        py::arg("database"), py::arg("num_sub_databases") = 1,
        py::arg("seed") = 0
    );

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
