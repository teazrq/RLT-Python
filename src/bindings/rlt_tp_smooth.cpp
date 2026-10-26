// Tensor-product penalized B-spline smoother for survival covariance
// smoothing. This is a Python alternative to RLT's index-coordinate
// mgcv::gam(z ~ te(i, j, k)), not a numerically equivalent port.
//
// The smoother fits z_{ab} = f(t_a, t_b) + e over the covariance matrix
// entries, with f a tensor-product of cubic B-spline bases in each
// coordinate, penalized by second differences and one shared smoothing
// parameter selected using an approximate GCV score.
//
// Layout note: covariance matrices here are timepoints x timepoints with
// irregular spacing; we map each coordinate through its knot sequence so
// the tensor basis reflects the actual time grid.

#include "RLT.h"
#include <vector>

using namespace arma;

namespace rlt {

// ---- cubic B-spline basis on arbitrary knots (Clamped) ----
// Returns basis matrix B (n x nbasis) and penalty D2 (nbasis x nbasis)
// from second differences of coefficient vectors.

static void bspline_basis(const vec& x, const vec& iknots, size_t degree,
                          mat& B, sp_mat* penalty_out = nullptr) {
  // extended knot sequence (clamped)
  std::vector<double> t;
  t.reserve(iknots.n_elem + 2 * degree);
  for (size_t i = 0; i < degree; i++) t.push_back(iknots(0));
  for (size_t i = 0; i < iknots.n_elem; i++) t.push_back(iknots(i));
  for (size_t i = 0; i < degree; i++) t.push_back(iknots(iknots.n_elem - 1));

  size_t nb = t.size() - degree - 1;  // number of basis functions

  B.set_size(x.n_elem, nb);
  B.zeros();

  // Cox-de Boor recursion evaluated directly (degree+1 nonzero per row)
  for (size_t pt = 0; pt < x.n_elem; pt++) {
    double xv = x(pt);
    // find span
    size_t k = 0;
    while (k < t.size() - degree - 2 && t[k + 1] <= xv) k++;
    // handle right endpoint
    if (xv >= t[t.size() - degree - 1]) k = t.size() - degree - 2;

    std::vector<double> B0(t.size() - 1, 0.0);
    // degree 0
    for (size_t i = 0; i < t.size() - 1; i++)
      B0[i] = (t[i] <= xv && xv < t[i + 1]) ? 1.0 : 0.0;
    // right endpoint closure
    if (xv == t.back())
      B0[std::max<int>((int)t.size() - (int)degree - 2, 0)] = 1.0;

    std::vector<double> Bk = B0;
    for (size_t d = 1; d <= degree; d++) {
      std::vector<double> Bn(t.size() - d - 1, 0.0);
      for (size_t i = 0; i < Bn.size(); i++) {
        double left = (t[i + d] > t[i])
                          ? Bk[i] * (xv - t[i]) / (t[i + d] - t[i])
                          : 0.0;
        double right =
            (t[i + d + 1] > t[i + 1])
                ? Bk[i + 1] * (t[i + d + 1] - xv) / (t[i + d + 1] - t[i + 1])
                : 0.0;
        Bn[i] = left + right;
      }
      Bk = Bn;
    }

    size_t off = k - degree;
    for (size_t j = 0; j <= degree; j++) {
      size_t idx = off + j;
      if (idx < nb) B(pt, idx) = Bk[k - degree + j];
    }
  }

  if (penalty_out) {
    // second-difference penalty on coefficients
    size_t m = nb;
    if (m >= 3) {
      mat D2(m - 2, m, fill::zeros);
      for (size_t i = 0; i < m - 2; i++) {
        D2(i, i) = 1.0;
        D2(i, i + 1) = -2.0;
        D2(i, i + 2) = 1.0;
    }
      sp_mat D2sp(D2);
      sp_mat P = D2sp.t() * D2sp;
      *penalty_out = P;
    } else {
      sp_mat P(nb, nb);
      *penalty_out = P;
    }
  }
}

}  // namespace rlt

// Fit the tensor-product smoother on an n x n covariance surface.
// Inputs:
//   Sigma (n x n) raw covariance; tvec (n) time points (may be irregular)
//   k (basis size per coordinate; mgcv te(i,j,k=K) uses K per margin)
//   lambda (smoothing parameter; if < 0, GCV-selected on a log grid)
// Returns smoothed, symmetrized matrix.
arma::mat rlt_tp_smooth(const arma::mat& Sigma, const arma::vec& tvec,
                        size_t k, double lambda_user) {
  using namespace rlt;

  size_t n = Sigma.n_rows;
  if (n < 4) return Sigma;

  size_t kk = std::max(k, (size_t)4);
  if (kk > n) kk = n;

  // marginal knots: quantile-based (like mgcv default knot placement)
  vec probs = linspace<vec>(0.0, 1.0, kk - 2);
  vec iknots = quantile(tvec, probs);
  // ensure strictly increasing
  iknots = unique(iknots);
  if (iknots.n_elem < 4) return Sigma;

  mat B;
  bspline_basis(tvec, iknots, 3, B);
  size_t m = B.n_cols;

  sp_mat P;
  bspline_basis(tvec, iknots, 3, B, &P);  // recompute with penalty
  // (bspline_basis computes both in one call when penalty_out given;
  //  second call kept for clarity of the interface)

  // Build the tensor design: rows = n^2 entries, cols = m^2
  // Z_{ab} = sum_{jl} c_{jl} B_a(j) B_b(l)
  // Vectorize: z = (B \kron B) c  with row-major over (a, b)
  // lambda penalizes each margin's roughness.

  vec z(n * n);
  for (size_t a = 0; a < n; a++)
    for (size_t b = 0; b < n; b++) z(a * n + b) = Sigma(a, b);

  // normal equations with Kronecker structure:
  // (B'B \kron B'B + lambda (P \kron I + I \kron P)) c = (B \kron B)' z
  mat BtB = B.t() * B;
  mat I_m = eye<mat>(m, m);
  mat Pdense(P);

  // response
  vec Bz(n * m);
  // (B kron B)^T z: first right-multiply rows, then columns
  // z as matrix Z (n x n): C_resp = B^T Z B  (m x m) then vectorize
  mat Zmat = Sigma;
  mat Cresp = B.t() * Zmat * B;
  vec cresp = vectorise(Cresp);

  double lambda = lambda_user;
  if (lambda < 0) {
    // GCV over a coarse log grid
    double best = 1e30;
    for (double lg = -6; lg <= 4.0; lg += 0.5) {
      double lam = std::pow(10.0, lg);
      mat A = kron(BtB, BtB) + lam * (kron(Pdense, I_m) + kron(I_m, Pdense));
      // solve and GCV
      vec c;
      bool ok = solve(c, A, cresp, solve_opts::likely_sympd + solve_opts::no_approx);
      if (!ok) continue;
      mat Chat = reshape(c, m, m);
      mat Shat = B * Chat * B.t();
      double rss = accu(square(Sigma - Shat));
      // effective df: trace of hat matrix (approx via fitted values method)
      // H = A^{-1} (BkronB)^T ; tr H = trace(A^{-1} kron-part) approx:
      // use edf = trace(BtB A^{-1} BtB) analog — expensive; approximate by
      // counting via residual variance profile: use simple GCV on rss only
      // with effective df estimated from eigenvalues of penalty
      vec eigP;
      eig_sym(eigP, Pdense);
      double edf = (double)(m * m) - accu(lam / (eigP + lam)) * m;
      double denom = 1.0 - edf / (double)(n * n);
      if (denom <= 0.05) continue;
      double gcv = (rss / (double)(n * n)) / (denom * denom);
      if (gcv < best) {
        best = gcv;
        lambda = lam;
      }
    }
    if (lambda < 0) lambda = 1.0;
  }

  mat A = kron(BtB, BtB) + lambda * (kron(Pdense, I_m) + kron(I_m, Pdense));
  vec c;
  bool ok = solve(c, A, cresp, solve_opts::likely_sympd + solve_opts::no_approx);
  if (!ok) {
    // ridge fallback
    A += 1e-8 * eye<mat>(A.n_rows, A.n_cols);
    solve(c, A, cresp, solve_opts::likely_sympd);
  }

  mat Chat = reshape(c, m, m);
  mat Shat = B * Chat * B.t();

  Shat = (Shat + Shat.t()) / 2.0;
  return Shat;
}
