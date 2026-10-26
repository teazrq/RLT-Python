"""Quantify same-R optimizer rounding using exact binary covariance inputs.

Run after generating and packing survival_smoother_mgcv_1_9_4.npz.
Usage: python generate_survival_sensitivity.py <output-directory>
Set RLT_RSCRIPT and RLT_REFERENCE_LIBRARY to the shared reference installs.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np

fixtures = Path(__file__).parent
arrays = np.load(fixtures / 'survival_smoother_mgcv_1_9_4.npz')
directory = Path(sys.argv[1]).resolve()
directory.mkdir(exist_ok=True)
tags = ['forest_lc1_IJ', 'forest_lc2_matched', 'forest_lc2_jack']
variants = ['base', 'reread', 'dense_up', 'dense_down', 'diagonal_up', 'pair_up']
for tag in tags:
    raw = arrays[tag + '_raw']
    values = dict(base=raw, reread=raw.copy(), dense_up=np.nextafter(raw, np.inf),
                  dense_down=np.nextafter(raw, -np.inf), diagonal_up=raw.copy(), pair_up=raw.copy())
    values['diagonal_up'][0, 0] = np.nextafter(raw[0, 0], np.inf)
    values['pair_up'][0, 1] = np.nextafter(raw[0, 1], np.inf)
    values['pair_up'][1, 0] = np.nextafter(raw[1, 0], np.inf)
    for name, value in values.items():
        (directory / f'{tag}_{name}.bin').write_bytes(
            np.asarray(value, dtype='<f8').ravel(order='F').tobytes())

script = f'''
reference <- Sys.getenv('RLT_REFERENCE_LIBRARY')
if(nzchar(reference)) .libPaths(c(reference,.libPaths()))
library(mgcv)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
root <- "{directory.as_posix()}"
read <- function(tag,variant) {{
 con <- file(file.path(root,paste0(tag,'_',variant,'.bin')),'rb');on.exit(close(con))
 matrix(readBin(con,what='double',n=49L,size=8L,endian='little'),7L,7L)
}}
surface <- function(fit,data) {{M<-matrix(predict(fit,newdata=data),7L,7L);(M+t(M))/2}}
for(tag in c({','.join(repr(tag) for tag in tags)})) {{
 G <- read(tag,'base');data <- expand.grid(i=1:7,j=1:7);data$z <- as.vector(G)
 base <- gam(z~te(i,j,k=c(6L,6L)),data=data);base_surface <- surface(base,data)
 base_fixed <- gam(z~te(i,j,k=c(6L,6L)),data=data,sp=base$sp)
 base_fixed_surface <- surface(base_fixed,data)
 for(variant in c({','.join(repr(variant) for variant in variants)})) {{
  input <- read(tag,variant);data$z <- as.vector(input)
  fit <- gam(z~te(i,j,k=c(6L,6L)),data=data)
  fixed <- gam(z~te(i,j,k=c(6L,6L)),data=data,sp=base$sp)
  fitted <- surface(fit,data);fixed_surface <- surface(fixed,data)
  conv <- fit$mgcv.conv
  values <- c(fit$sp,sum(fit$edf),fit$gcv.ubre,conv$iter,
    max(abs(fitted-base_surface))/max(abs(G)),max(abs(input-G))/max(abs(G)),
    max(abs(fixed_surface-base_fixed_surface))/max(abs(G)),conv$rms.grad,conv$rank,
    conv$fully.converged)
  writeLines(sprintf('%.17g',values),file.path(root,paste0(tag,'_',variant,'_metadata.txt')))
 }}
}}
'''
source = directory / 'ulp.R'
source.write_text(script)
rscript = os.environ.get('RLT_RSCRIPT') or shutil.which('Rscript')
if not rscript:
    raise RuntimeError('Set RLT_RSCRIPT to a shared Rscript installation')
subprocess.run([rscript, '--vanilla', str(source)], check=True, timeout=60)
records = []
for tag in tags:
    for variant in variants:
        v = np.loadtxt(directory / f'{tag}_{variant}_metadata.txt')
        records.append(dict(case=tag, variant=variant, sp=v[:2].tolist(), edf=float(v[2]),
            gcv=float(v[3]), iterations=int(v[4]), optimized_surface_delta_scaled=float(v[5]),
            input_delta_scaled=float(v[6]), fixed_sp_surface_delta_scaled=float(v[7]),
            rms_gradient=float(v[8]), rank=int(v[9]), fully_converged=bool(v[10])))
manifest_path = fixtures / 'survival_smoother_mgcv_1_9_4.json'
manifest = json.loads(manifest_path.read_text())
manifest['optimizer_roundoff_controls'] = records
manifest['raw_input_format'] = 'R writeBin IEEE754 little-endian binary doubles'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
