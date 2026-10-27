# Regenerate the version-pinned smoother oracle with shared R packages.
# Rscript generate_survival_smoother.R <output-directory>
# R_LIBS_USER should identify the corrected RLT reference installation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L, dir.exists(args[1]))
out <- args[1]
script <- grep('^--file=',commandArgs(),value=TRUE)
stopifnot(length(script)==1L)
root <- dirname(dirname(dirname(normalizePath(sub('^--file=','',script),mustWork=TRUE))))
manifest <- readLines(file.path(root,'tools','r-reference.json'),warn=FALSE)
manifest_string <- function(name) {
  pattern <- paste0('^\\s*"',name,'"\\s*:\\s*"([^"]+)".*$')
  line <- grep(pattern,manifest,value=TRUE)
  stopifnot(length(line)==1L)
  sub(pattern,'\\1',line)
}
reference_revision <- Sys.getenv('RLT_REFERENCE_REVISION')
if (!nzchar(reference_revision) || reference_revision!=manifest_string('revision'))
  stop('Set RLT_REFERENCE_REVISION to the pinned tools/r-reference.json revision after verifying and installing that checkout')
reference_library <- Sys.getenv('RLT_REFERENCE_LIBRARY')
if (nzchar(reference_library)) .libPaths(c(reference_library,.libPaths()))
library(mgcv)
stopifnot(as.character(packageVersion('mgcv'))=='1.9.4')
library(RLT)
stopifnot(as.character(packageVersion('RLT'))==manifest_string('package_version'))
reference_dll <- getLoadedDLLs()[['RLT']][['path']]
stopifnot(length(reference_dll)==1L, file.exists(reference_dll))
reference_dll <- normalizePath(reference_dll,mustWork=TRUE)
reference_md5 <- unname(tools::md5sum(reference_dll))
stopifnot(length(reference_md5)==1L, !is.na(reference_md5))
message('Loaded RLT native library: ',reference_dll)
write_numeric <- function(x,file) {
  # write.csv writes about 15 significant digits. A 17-digit round trip is
  # necessary when a GCV step compares scores at machine precision.
  x <- as.data.frame(x)
  x[] <- lapply(x,function(column) sprintf('%.17g',column))
  write.table(x,file,sep=',',row.names=FALSE,col.names=TRUE,quote=FALSE)
}
cases <- list()
add <- function(id,G,k) cases[[id]] <<- list(G=G,k=k)
toeplitz_cov <- function(n) outer(seq_len(n),seq_len(n),function(i,j) exp(-abs(i-j)/2))+
  diag(seq(.01,.04,length.out=n))
for (n in c(3L,4L,5L,6L,12L)) for (k in unique(pmin(c(0L,2L,3L,4L,6L,10L),n)))
  add(paste0('toeplitz_n',n,'_k',k),toeplitz_cov(n),k)
for (scale in c(1e-12,1e-6,1e6)) add(paste0('scale_',format(scale,scientific=TRUE)),scale*toeplitz_cov(12),6L)
t <- seq_len(12)
add('zero',matrix(0,12,12),6L)
add('constant',matrix(.3,12,12),6L)
v <- 1+sin(t/2)
add('rank_one',outer(v,v),6L)
F <- cbind(sin(t),cos(t),t/12)
add('rank_three',tcrossprod(F),6L)
add('diagonal',diag(seq(.1,1,length.out=12)),6L)
add('brownian',outer(t,t,pmin)/12,6L)
Q <- qr.Q(qr(cbind(rep(1,12),outer(t,1:11,function(a,b)cos(a*b)))))
add('ill_conditioned',Q%*%diag(10^seq(0,-12,length.out=12))%*%t(Q),6L)

u <- seq_len(48)
X <- cbind(sin(u*.7),cos(u*.31),((u*7)%%19)/19)
y <- 1+((u*11+floor(8*X[,1]))%%23)
event <- as.integer(u%%4!=0)
for (lc in c(1L,2L)) for (mode in c('matched','IJ','jack')) {
  fit <- RLT(X,y,censor=event,model='survival',ntrees=24,nmin=3,mtry=2,
    linear.comb=lc,linear.comb.method=if(lc>1L)'naive' else 'default',
    var.mode=if(mode=='matched')'matched' else 'IJ',seed=37,ncores=1)
  pred <- predict(fit,X[1:2,,drop=FALSE],var.est=TRUE,var.mode=mode,band.grid.size=7)
  add(paste0('forest_lc',lc,'_',mode),pred$Cov[,,1],6L)
}
# Independent holdout definitions were frozen in audit before choosing a
# sensitivity-aware optimizer acceptance gate (SHA256 8204976a122982f6d44cc67679ea75ef909b91b6050bd39fd2294a4094698f8f).
for(n in c(7L,9L,11L,15L)) {
  t <- seq_len(n)
  add(paste0('holdout_exponential_n',n),
    outer(t,t,function(a,b)exp(-abs(a-b)/3.7))+diag(seq(.02,.07,length.out=n)),min(6L,n))
  set.seed(8900+n)
  F <- matrix(rnorm(n*4),n,4)
  add(paste0('holdout_lowrank_n',n),tcrossprod(F)/4,min(4L,n))
  set.seed(9200+n)
  F <- matrix(rnorm(n*n),n,n)
  add(paste0('holdout_fullrank_n',n),tcrossprod(F)/n,min(8L,n))
}
metadata <- list()
for (id in names(cases)) {
  G <- cases[[id]]$G;n <- nrow(G);ks <- min(max(cases[[id]]$k,3),n)
  data <- expand.grid(i=seq_len(n),j=seq_len(n));data$z <- as.vector(G)
  write_numeric(G,file.path(out,paste0(id,'_raw.csv')))
  writeBin(as.double(G),file.path(out,paste0(id,'_raw.bin')),size=8,endian='little')
  record <- data.frame(id=id,n=n,k=cases[[id]]$k,status='ok',sp1=NA_real_,sp2=NA_real_,
    edf=NA_real_,gcv=NA_real_,iterations=NA_integer_,converged=NA,rank=NA_integer_,
    rms_gradient=NA_real_,error='')
  tryCatch({
    model <- gam(z~te(i,j,k=c(ks,ks)),data=data)
    surface <- matrix(predict(model,newdata=data),n,n);surface <- (surface+t(surface))/2
    write_numeric(surface,file.path(out,paste0(id,'_smooth.csv')))
    pred <- structure(list(CHF=matrix(seq(.1,1,length.out=n),1),
      Cov=array(G,c(n,n,1)),timepoints=seq_len(n)^3),class=c('RLT','pred','surv'))
    band <- get.surv.band(pred,approach='smoothed',k_rank=cases[[id]]$k,nsim=2)
    write_numeric(band[[1]]$diagnostics$diag_sd,file.path(out,paste0(id,'_band_sd.csv')))
    record$sp1 <- model$sp[1];record$sp2 <- model$sp[2];record$edf <- sum(model$edf)
    record$gcv <- model$gcv.ubre;record$iterations <- model$mgcv.conv$iter
    record$converged <- model$mgcv.conv$fully.converged
    record$rank <- model$mgcv.conv$rank;record$rms_gradient <- model$mgcv.conv$rms.grad
  },error=function(e) {record$status <<- 'error';record$error <<- conditionMessage(e)})
  metadata[[id]] <- record
}
metadata <- do.call(rbind,metadata)
metadata[] <- lapply(metadata,function(column) if(is.numeric(column)) sprintf('%.17g',column) else column)
write.csv(metadata,file.path(out,'metadata.csv'),row.names=FALSE)
G <- toeplitz_cov(12);data <- expand.grid(i=1:12,j=1:12);data$z <- as.vector(G)
fixed <- gam(z~te(i,j,k=c(6L,6L)),data=data,sp=c(.1,.3))
write_numeric(matrix(predict(fixed,newdata=data),12,12),file.path(out,'fixed_sp_surface.csv'))
setup <- gam(z~te(i,j,k=c(6L,6L)),data=data,fit=FALSE)
write_numeric(setup$X,file.path(out,'design.csv'))
for (j in seq_along(setup$S)) write_numeric(setup$S[[j]],file.path(out,paste0('penalty',j,'.csv')))
set.seed(214)
factor <- matrix(c(2,.5,.1,0,sqrt(.75),.2,0,0,sqrt(.2)),3,3)
draws <- factor%*%matrix(rnorm(3*17),3,17)
sd <- sqrt(rowSums(factor^2))
alpha <- c(.05,.1,.5)
maxima <- apply(abs(draws/sd),2,max)
widths <- sd%o%as.numeric(quantile(maxima,1-alpha,type=5))
write_numeric(draws,file.path(out,'mc_draws.csv'))
write_numeric(t(widths),file.path(out,'mc_widths.csv'))
write_numeric(sd,file.path(out,'mc_scales.csv'))
writeLines(c(paste('R',getRversion()),paste('mgcv',packageVersion('mgcv')),
  paste('RLT asserted source revision',reference_revision),
  paste('RLT',packageVersion('RLT')),paste('RLT loaded native library',basename(reference_dll)),
  paste('Platform',R.version$platform),
  paste('RLT DLL MD5',reference_md5)),file.path(out,'versions.txt'))
