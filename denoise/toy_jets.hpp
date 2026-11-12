// toy_jets.hpp
//
// Toy heavy-ion jet images shared by the jet test programs in this folder.
//
// Each event is a square (eta, phi) image centred on a jet axis, pixel value =
// summed pT [GeV] of the particles landing in that pixel:
//   clean = one toy jet (collimated spray of constituents inside R)
//   noisy = clean + thermal background (same model as generate_samples.py:
//           uniform in eta-phi, pT ~ Gamma(shape 2, mean 1.2 GeV),
//           15000 particles over |eta| < 3)

#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace toy {

struct Config {
    int nEvents = 500;
    unsigned seed = 1;
    int npix = 64;             // image is npix x npix
    double halfWidth = 1.6;    // image spans [-halfWidth, halfWidth] in eta and phi
    double jetPt = 100.0;      // GeV
    double R = 0.4;            // jet radius
    double bkgDensity = 15000.0 / (6.0 * 2.0 * CV_PI);  // particles per unit eta-phi area
    double bkgMeanPt = 1.2;    // GeV

    double pixelSize() const { return 2.0 * halfWidth / npix; }
    double pixelArea() const { return pixelSize() * pixelSize(); }
};

// Add pt to the pixel containing (eta, phi); particles outside the image are dropped.
inline void deposit(cv::Mat1f& img, const Config& c, double eta, double phi, double pt) {
    int col = static_cast<int>(std::floor((eta + c.halfWidth) / c.pixelSize()));
    int row = static_cast<int>(std::floor((phi + c.halfWidth) / c.pixelSize()));
    if (col < 0 || col >= c.npix || row < 0 || row >= c.npix) return;
    img(row, col) += static_cast<float>(pt);
}

// Toy jet: 8 + Poisson(8) constituents, exponential momentum sharing,
// radial profile peaked at the axis (r = 0.9 R u^2).
inline cv::Mat1f makeJet(const Config& c, std::mt19937& rng) {
    cv::Mat1f img = cv::Mat1f::zeros(c.npix, c.npix);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::exponential_distribution<double> expo(1.0);
    int n = 8 + std::poisson_distribution<int>(8.0)(rng);

    std::vector<double> w(n);
    double sum = 0.0;
    for (double& x : w) { x = expo(rng); sum += x; }
    for (double x : w) {
        double u = uni(rng);
        double r = 0.9 * c.R * u * u;
        double a = 2.0 * CV_PI * uni(rng);
        deposit(img, c, r * std::cos(a), r * std::sin(a), c.jetPt * x / sum);
    }
    return img;
}

inline cv::Mat1f makeBackground(const Config& c, std::mt19937& rng) {
    cv::Mat1f img = cv::Mat1f::zeros(c.npix, c.npix);
    double area = 4.0 * c.halfWidth * c.halfWidth;
    int n = std::poisson_distribution<int>(c.bkgDensity * area)(rng);
    std::uniform_real_distribution<double> pos(-c.halfWidth, c.halfWidth);
    std::gamma_distribution<double> pt(2.0, c.bkgMeanPt / 2.0);
    for (int i = 0; i < n; ++i) {
        double eta = pos(rng), phi = pos(rng);
        deposit(img, c, eta, phi, pt(rng));
    }
    return img;
}

// Distance of each pixel centre from the jet axis (the image centre).
inline cv::Mat1f radiusMap(const Config& c) {
    cv::Mat1f radius(c.npix, c.npix);
    for (int r = 0; r < c.npix; ++r)
        for (int col = 0; col < c.npix; ++col) {
            double eta = -c.halfWidth + (col + 0.5) * c.pixelSize();
            double phi = -c.halfWidth + (r + 0.5) * c.pixelSize();
            radius(r, col) = static_cast<float>(std::hypot(eta, phi));
        }
    return radius;
}

// Cone mask: 1 for pixels whose centre is within R of the jet axis, else 0.
inline cv::Mat1f coneMask(const Config& c) {
    cv::Mat1f cone = radiusMap(c);
    for (float& v : cone) v = v < c.R ? 1.0f : 0.0f;
    return cone;
}

// One labelled picture tile: values clipped to [0, vmax], sqrt colour scale.
inline cv::Mat tile(const cv::Mat1f& img, double vmax, const std::string& label) {
    cv::Mat1f s;
    cv::max(img, 0.0, s);
    cv::min(s, vmax, s);
    cv::sqrt(s / vmax, s);
    cv::Mat u8, colour;
    s.convertTo(u8, CV_8U, 255.0);
    cv::applyColorMap(u8, colour, cv::COLORMAP_INFERNO);
    cv::resize(colour, colour, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);
    cv::putText(colour, label, cv::Point(6, 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return colour;
}

}  // namespace toy
