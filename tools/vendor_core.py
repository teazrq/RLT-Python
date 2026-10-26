#!/usr/bin/env python3
"""Vendor the RLT C++ core into src/cpp/, stripping R/Rcpp dependencies.

Idempotent: writes only changed files from an explicit R package source.
An optional destination allows reviewing generated files before updating src/cpp/.
Excluded (by design):
  - quantile/   : out of scope for rlt-python
  - r-interface/: replaced by the pybind11 layer, except core c-index helpers
  - kernels/    : deferred to the kernel milestone (Rcpp-typed signatures)
  - RLT_init.cpp, Makevars*: R build machinery

Transformations:
  - #include <RcppArmadillo.h>  -> #include <armadillo> + "rlt_compat.h"
  - using namespace Rcpp;       -> removed
  - dqrng / boost::random       -> rlt:: vendored RNG (via compat header)
  - Rcpp::warning               -> rlt_warning
  - PARAM_READ_R(Rcpp List)     -> PARAM_READ(rlt::CoreParams const&)
  - RLTcout macro               -> std::cout
"""
import argparse
import re
import sys
from pathlib import Path

DST = Path(__file__).resolve().parents[1] / "src" / "cpp"

EXCLUDE_DIRS = {"quantile", "r-interface", "kernels", "include/quantile"}
EXCLUDE_FILES = {"RLT_init.cpp", "Makevars", "Makevars.win", "RcppExports.cpp"}

# ---------------------------------------------------------------- compat header
RLT_COMPAT = r'''// ************************************************************
// rlt-python compatibility header
// Replaces Rcpp/R/dqrng/boost dependencies of the original RLT
// core with self-contained, deterministic implementations.
// ************************************************************
#ifndef RLT_COMPAT_H
#define RLT_COMPAT_H

#include <armadillo>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

namespace rlt {

// ---------- deterministic RNG: xoshiro256plus ----------
class xoshiro256plus {
  std::uint64_t s[4];

  static std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  static std::uint64_t rotl(const std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

 public:
  xoshiro256plus() : s{1, 0, 0, 0} {}
  explicit xoshiro256plus(std::uint64_t seed) {
    std::uint64_t sm = seed;
    for (auto& v : s) v = splitmix64(sm);
    if (!(s[0] | s[1] | s[2] | s[3])) s[0] = 1;  // never all-zero
  }

  std::uint64_t next() {
    const std::uint64_t result = s[0] + s[3];
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }

  double uniform01() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

// ---------- boost::random-exact distributions ----------
// Bit-exact replicas of boost::random::uniform_int_distribution and
// uniform_real_distribution as consumed by the original RLT core with a
// 64-bit integral engine (dqrng xoshiro256plus: min=0, max=2^64-1).
// This is what makes same-seed fits identical to the R package.

template <typename T>
class uniform_int_distribution {
 public:
  using result_type = T;

  uniform_int_distribution(T min_v, T max_v)
      : min_(min_v),
        range_(static_cast<std::uint64_t>(max_v) -
               static_cast<std::uint64_t>(min_v)) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    // boost::random::detail::generate_uniform_int, integral-engine branch
    const std::uint64_t brange = ~std::uint64_t(0);  // eng.max - eng.min
    if (range_ == 0) return min_;  // boost draws NOTHING here
    std::uint64_t result;
    if (brange == range_) {
      result = rng.next();
    } else {
      // brange == numeric_limits<base_unsigned>::max() branch:
      std::uint64_t bucket_size = brange / (range_ + 1);
      if (brange % (range_ + 1) == range_) ++bucket_size;
      for (;;) {
        result = rng.next() / bucket_size;
        if (result <= range_) break;
      }
    }
    return static_cast<T>(result + min_);
  }

 private:
  T min_;
  std::uint64_t range_;
};

// ---------- uniform reals [min, max] ----------
template <typename T>
class uniform_real_distribution {
 public:
  using result_type = T;

  uniform_real_distribution(T min_v, T max_v) : min_(min_v), max_(max_v) {}

  template <typename RNG>
  T operator()(RNG& rng) const {
    // boost::random::detail::generate_uniform_real, integral-engine branch:
    // numerator / (double)(eng.max - eng.min + 1) * span + min, retry if
    // the rounding hits the upper bound.
    const T divisor = static_cast<T>(~std::uint64_t(0)) + 1.0;  // 2^64
    for (;;) {
      T numerator = static_cast<T>(rng.next());
      T result = numerator / divisor * (max_ - min_) + min_;
      if (result < max_) return result;
    }
  }

 private:
  T min_, max_;
};

// ---------- plain parameter struct (replaces the Rcpp List) ----------
struct CoreParams {
  // main parameters
  size_t n = 0;
  size_t p = 0;
  size_t ntrees = 1;
  size_t mtry = 1;
  size_t nmin = 1;
  size_t nsplit = 1;
  bool replacement = false;
  double resample_prob = 0.8;
  bool use_obs_w = false;
  bool use_var_prob = false;
  size_t importance = 0;
  bool reinforcement = false;
  // other control parameters
  bool obs_track = false;
  size_t var_mode = 0;
  size_t linear_comb = 1;
  double alpha = 0;
  size_t split_rule = 1;
  size_t linear_comb_method = 1;
  // embedded (reinforcement) forest parameters
  size_t embed_ntrees = 0;
  double embed_mtry = 0;
  size_t embed_nmin = 0;
  size_t embed_nsplit = 0;
  bool embed_replacement = false;
  double embed_resample_prob = 0;
  double embed_mute = 0;
  size_t embed_protect = 0;
  double embed_threshold = 0.25;
  // system
  size_t ncores = 1;
  size_t verbose = 0;
  size_t seed = 1;
};

}  // namespace rlt

// warning shim: routes through std::cerr (Python layer maps to warnings)
inline void rlt_warning(const std::string& msg) {
  std::cerr << "RLT warning: " << msg << std::endl;
}

// printf-style shim for the couple of Rprintf() sites in the core
#include <cstdarg>
#include <cstdio>
inline void Rprintf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "\n");
}

#endif  // RLT_COMPAT_H
'''

PARAM_READ_NEW = r'''  void PARAM_READ(const rlt::CoreParams& q) {
    N = q.n;                    P = q.p;
    ntrees = q.ntrees;          mtry = q.mtry;
    nmin = q.nmin;              nsplit = q.nsplit;
    replacement = q.replacement;
    resample_prob = q.resample_prob;
    useobsweight = q.use_obs_w; usevarprob = q.use_var_prob;
    importance = q.importance;  reinforcement = q.reinforcement;
    obs_track = q.obs_track;    var_mode = q.var_mode;
    linear_comb = q.linear_comb; alpha = q.alpha;
    split_rule = q.split_rule;
    linear_comb_method = q.linear_comb_method;
    embed_ntrees = q.embed_ntrees;        embed_mtry = q.embed_mtry;
    embed_nmin = q.embed_nmin;            embed_nsplit = q.embed_nsplit;
    embed_replacement = q.embed_replacement;
    embed_resample_prob = q.embed_resample_prob;
    embed_mute = q.embed_mute;            embed_protect = q.embed_protect;
    embed_threshold = q.embed_threshold;
    ncores = q.ncores;          verbose = q.verbose;
    seed = q.seed;
  };'''


RLT_H = r'''//    ----------------------------------------------------------------
//
//    Reinforcement Learning Trees (RLT) - python port main header
//    Core + regression + classification + survival
//    (quantile module intentionally excluded from rlt-python)
//
//    ----------------------------------------------------------------

#ifndef RLT_H
#define RLT_H

#include "rlt_compat.h"

// Core infrastructure
#include "include/core/Utility.h"
#include "include/core/Tree_Definition.h"
#include "include/core/Tree_Function.h"
#include "include/core/Stat_Function.h"
#include "include/core/Forest_Build_Guards.h"
#include "include/core/Variance_IJ_Jack.h"

// Regression
#include "include/regression/Reg_Uni_Definition.h"
#include "include/regression/Reg_Uni_Function.h"

// Classification
#include "include/classification/Cla_Uni_Definition.h"
#include "include/classification/Cla_Uni_Function.h"

// Survival
#include "include/survival/Surv_Uni_Definition.h"
#include "include/survival/Surv_Uni_Function.h"

#endif  // RLT_H
'''


def splice_method(text: str, header: str, replacement: str) -> str:
    """Replace the brace-balanced block starting at `header` (a signature
    prefix) with `replacement` (which must itself be balanced)."""
    start = text.index(header)
    brace_open = text.index("{", start)
    depth, i = 0, brace_open
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    end = i + 1
    # consume trailing semicolon if present
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == ";":
        end += 1
    return text[:start] + replacement + text[end:]


def transform(text: str, path: Path) -> str:
    # 1. RcppArmadillo -> armadillo + compat
    text = re.sub(r'#\s*include\s*<RcppArmadillo\.h>',
                  '#include <armadillo>\n#include "rlt_compat.h"', text)
    # 2. drop R namespace + dqrng/boost includes
    text = re.sub(r'^\s*using\s+namespace\s+Rcpp\s*;\s*\n', '', text,
                  flags=re.M)
    text = re.sub(r'#\s*include\s*<dqrng[^>]*>\s*\n?', '', text)
    text = re.sub(r'#\s*include\s*<xoshiro[^>]*>\s*\n?', '', text)
    text = re.sub(r'#\s*include\s*<boost[^>]*>\s*\n?', '', text)
    # 3. RNG swaps
    text = text.replace("dqrng::xoshiro256plus", "rlt::xoshiro256plus")
    text = text.replace("boost::random::uniform_int_distribution",
                        "rlt::uniform_int_distribution")
    text = text.replace("boost::random::uniform_real_distribution",
                        "rlt::uniform_real_distribution")
    # 4. warnings
    text = text.replace("Rcpp::warning", "rlt_warning")
    # 4b. R logical literals
    text = re.sub(r'\bTRUE\b', 'true', text)
    text = re.sub(r'\bFALSE\b', 'false', text)
    # 4c. drop R-facing entry-point declarations (List XxxForestFit(...);)
    #     their implementations live in the excluded r-interface layer;
    #     the pybind11 layer declares its own typed entry points.
    text = re.sub(r'\nList\s+\w+\([^;]*?\);\n', '\n', text, flags=re.S)
    text = re.sub(r'^\s*//\s*\[\[Rcpp::export\(\)\]\]\s*\n', '', text,
                  flags=re.M)
    # 5. RLTcout -> std::cout (defined once in Utility.h)
    text = text.replace("#define RLTcout Rcout", "#define RLTcout std::cout")
    # 6. Rcpp::as / Rf_length sites (PARAM_READ_R) — spliced below
    return text


def generate(source: Path) -> tuple[dict[str, str], list[str]]:
    """Transform sources in memory before writing any destination files."""
    generated, skipped = {}, []
    for p in sorted(source.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(source).as_posix()
        if any(rel == d or rel.startswith(d + "/") for d in EXCLUDE_DIRS):
            skipped.append(rel)
            continue
        if p.name in EXCLUDE_FILES:
            skipped.append(rel)
            continue
        if p.suffix not in {".h", ".cpp"}:
            continue
        generated[rel] = transform(p.read_text(encoding="utf-8"), p)

    # --- Utility.h specials: PARAM_READ_R splice ---
    utility = "include/core/Utility.h"
    generated[utility] = splice_method(generated[utility],
        "void PARAM_READ_R(List& param)", PARAM_READ_NEW)
    generated["core/Stat_Function.cpp"] = generated["core/Stat_Function.cpp"].rstrip() + "\n"

    # The survival core calls these helpers from the otherwise excluded
    # R interface. Use the actual source so fixes cannot leave a stale copy.
    cindex = source / "r-interface" / "cindex.cpp"
    cindex_text = transform(cindex.read_text(encoding="utf-8"), cindex).rstrip()
    generated["core/cindex.cpp"] = "\n".join(line.rstrip() for line in cindex_text.splitlines()) + "\n"

    # --- top-level headers ---
    generated["rlt_compat.h"] = RLT_COMPAT
    generated["RLT.h"] = RLT_H
    return generated, skipped


def residual_references(generated: dict[str, str]) -> dict[str, list[str]]:
    """Report R dependencies in code, ignoring compatibility comments."""
    bad = re.compile(r"Rcpp::|RcppExport|\bRcout\b|\bRf_|\bSEXP\b|"
                     r"RcppArmadillo|\bdqrng\b|boost::random|"
                     r"R Inside|R_inside")
    residuals = {}
    for rel, text in generated.items():
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
        for n, line in enumerate(code.splitlines(), 1):
            if bad.search(line):
                residuals.setdefault(rel, []).append(f"{n}: {line.strip()}")
    return residuals


def write_changed(path: Path, text: str) -> bool:
    """Keep untouched files and preserve the newline style of changed files."""
    newline = "\n"
    if path.exists():
        previous = path.read_bytes()
        if previous.decode("utf-8").replace("\r\n", "\n") == text:
            return False
        if b"\r\n" in previous:
            newline = "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline=newline)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RLT R-package src directory")
    parser.add_argument("--dest", type=Path, default=DST,
                        help="candidate destination (default: src/cpp)")
    args = parser.parse_args(argv)
    source, destination = args.source.resolve(), args.dest.resolve()
    if not (source / "include/core/Utility.h").is_file():
        parser.error(f"not an RLT source directory: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("source and destination directories must not overlap")

    generated, skipped = generate(source)
    residuals = residual_references(generated)
    if residuals:
        print("Residual R/Rcpp references:", file=sys.stderr)
        for f, lines in residuals.items():
            print(f"{f}: " + "; ".join(lines), file=sys.stderr)
        return 1

    changed = sum(write_changed(destination / rel, text)
                  for rel, text in generated.items())
    print(f"vendored {len(generated)} files -> {destination} ({changed} changed)")
    print(f"skipped {len(skipped)} files (by design); no residual R/Rcpp references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
