// Python bindings for jetdenoise.hpp. Build with build.sh.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jetdenoise.hpp"

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
namespace jd = jetdenoise;

static void check(const Arr& a, const Arr& b, const Arr& c) {
  if (a.size() != b.size() || a.size() != c.size())
    throw std::invalid_argument("pt, eta, phi must have the same length");
}

static py::array_t<double> to_np(const std::vector<double>& v) {
  py::array_t<double> out(v.size());
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

static py::array_t<double> img_np(const jd::Image& im) {
  py::array_t<double> out({im.ne, im.np});
  std::copy(im.v.begin(), im.v.end(), out.mutable_data());
  return out;
}

PYBIND11_MODULE(jetdenoise, m) {
  m.doc() = "Heavy-ion background estimation and denoising (constituent subtraction, starlet wavelets)";

  py::class_<jd::BackgroundMap>(m, "BackgroundMap")
      .def_readonly("a0", &jd::BackgroundMap::a0, "mean rho [GeV per unit area]")
      .def_readonly("an", &jd::BackgroundMap::an)
      .def_readonly("bn", &jd::BackgroundMap::bn)
      .def("rho", [](const jd::BackgroundMap& b, double phi) { return b(phi); })
      .def("vn", &jd::BackgroundMap::vn);

  m.def("background_map",
        [](Arr pt, Arr eta, Arr phi, double eta_max, int n_eta, int n_phi, int nmax) {
          check(pt, eta, phi);
          py::gil_scoped_release nogil;
          return jd::background_map(pt.data(), eta.data(), phi.data(), pt.size(), eta_max, n_eta,
                                    n_phi, nmax);
        },
        py::arg("pt"), py::arg("eta"), py::arg("phi"), py::arg("eta_max") = 3.0,
        py::arg("n_eta") = 12, py::arg("n_phi") = 12, py::arg("nmax") = 3);

  m.def("constituent_subtract",
        [](Arr pt, Arr eta, Arr phi, const jd::BackgroundMap& bkg, double eta_max,
           double ghost_size, double max_dr, double alpha) {
          check(pt, eta, phi);
          std::vector<double> r;
          {
            py::gil_scoped_release nogil;
            r = jd::constituent_subtract(pt.data(), eta.data(), phi.data(), pt.size(), bkg,
                                         eta_max, ghost_size, max_dr, alpha);
          }
          return to_np(r);
        },
        py::arg("pt"), py::arg("eta"), py::arg("phi"), py::arg("bkg"), py::arg("eta_max") = 3.0,
        py::arg("ghost_size") = 0.1, py::arg("max_dr") = 0.25, py::arg("alpha") = 0.0);

  m.def("background_spectrum",
        [](Arr pt, Arr eta, Arr phi, std::vector<double> axes_eta, std::vector<double> axes_phi,
           std::vector<double> pt_edges, const jd::BackgroundMap& bkg, double eta_max,
           double exclude_r) {
          check(pt, eta, phi);
          std::vector<double> r;
          {
            py::gil_scoped_release nogil;
            r = jd::background_spectrum(pt.data(), eta.data(), phi.data(), pt.size(), axes_eta,
                                        axes_phi, pt_edges, bkg, eta_max, exclude_r);
          }
          return to_np(r);
        },
        py::arg("pt"), py::arg("eta"), py::arg("phi"), py::arg("axes_eta"), py::arg("axes_phi"),
        py::arg("pt_edges"), py::arg("bkg"), py::arg("eta_max") = 3.0, py::arg("exclude_r") = 0.8);

  m.def("posterior_weights",
        [](Arr pt, Arr eta, Arr phi, std::vector<double> axes_eta, std::vector<double> axes_phi,
           std::vector<double> pt_edges, std::vector<double> r_edges, Arr sig_template,
           std::vector<double> bkg_spectrum, const jd::BackgroundMap& bkg) {
          check(pt, eta, phi);
          std::vector<double> tmpl(sig_template.data(), sig_template.data() + sig_template.size());
          std::vector<double> r;
          {
            py::gil_scoped_release nogil;
            r = jd::posterior_weights(pt.data(), eta.data(), phi.data(), pt.size(), axes_eta,
                                      axes_phi, pt_edges, r_edges, tmpl, bkg_spectrum, bkg);
          }
          return to_np(r);
        },
        py::arg("pt"), py::arg("eta"), py::arg("phi"), py::arg("axes_eta"), py::arg("axes_phi"),
        py::arg("pt_edges"), py::arg("r_edges"), py::arg("sig_template"), py::arg("bkg_spectrum"),
        py::arg("bkg"));

  m.def("starlet_denoise",
        [](Arr pt, Arr eta, Arr phi, const jd::BackgroundMap& bkg, double eta_max, int n_eta,
           int n_phi, int n_scales, double k_sigma, bool soft, int carry_levels, bool return_images) {
          check(pt, eta, phi);
          jd::StarletResult r;
          {
            py::gil_scoped_release nogil;
            r = jd::starlet_denoise(pt.data(), eta.data(), phi.data(), pt.size(), bkg, eta_max,
                                    n_eta, n_phi, n_scales, k_sigma, soft, carry_levels);
          }
          py::dict out;
          out["weight"] = to_np(r.weight);
          out["sigma"] = to_np(r.sigma);
          if (return_images) {
            out["input"] = img_np(r.input);
            out["signal"] = img_np(r.signal);
            out["bkg"] = img_np(r.bkg);
          }
          return out;
        },
        py::arg("pt"), py::arg("eta"), py::arg("phi"), py::arg("bkg"), py::arg("eta_max") = 3.0,
        py::arg("n_eta") = 120, py::arg("n_phi") = 126, py::arg("n_scales") = 5,
        py::arg("k_sigma") = 3.0, py::arg("soft") = false, py::arg("carry_levels") = 3,
        py::arg("return_images") = false);
}
