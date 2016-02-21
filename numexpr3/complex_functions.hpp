#ifndef NUMEXPR_COMPLEX_FUNCTIONS_HPP
#define NUMEXPR_COMPLEX_FUNCTIONS_HPP

/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// TODO: Could just use std::complex<float> and std::complex<double>

/* constants */
static npy_cdouble nc_1 = {1., 0.};
static npy_cdouble nc_half = {0.5, 0.};
static npy_cdouble nc_i = {0., 1.};
static npy_cdouble nc_i2 = {0., 0.5};
/*
static npy_cdouble nc_mi = {0., -1.};
static npy_cdouble nc_pi2 = {M_PI/2., 0.};
*/

/* *************************** WARNING *****************************
Due to the way Numexpr places the results of operations, the *x and *r
pointers do point to the same address (apparently this doesn't happen
in NumPy).  So, measures should be taken so as to not to reuse *x
after the first *r has been overwritten.
*********************************************************************
*/

static void
nc_assign(npy_cdouble *x, npy_cdouble *r)
{
  r->real = x->real;
  r->imag = x->imag;
  return;
}

static void
nc_sum(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r)
{
    r->real = a->real + b->real;
    r->imag = a->imag + b->imag;
    return;
}

static void
nc_diff(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r)
{
    r->real = a->real - b->real;
    r->imag = a->imag - b->imag;
    return;
}

static void
nc_neg(npy_cdouble *a, npy_cdouble *r)
{
    r->real = -a->real;
    r->imag = -a->imag;
    return;
}

static void
nc_conj(npy_cdouble *a, npy_cdouble *r)
{
    r->real = a->real;
    r->imag = -a->imag;
    return;
}

// Needed for allowing the internal casting in numexpr machinery for
// conjugate operations
inline double fconj(double x)
{
    return x;
}

static void
nc_prod(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r)
{
    double ar=a->real, br=b->real, ai=a->imag, bi=b->imag;
    r->real = ar*br - ai*bi;
    r->imag = ar*bi + ai*br;
    return;
}

static void
nc_quot(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r)
{
    double ar=a->real, br=b->real, ai=a->imag, bi=b->imag;
    double d = br*br + bi*bi;
    r->real = (ar*br + ai*bi)/d;
    r->imag = (ai*br - ar*bi)/d;
    return;
}

static void
nc_sqrt(npy_cdouble *x, npy_cdouble *r)
{
    double s,d;
    if (x->real == 0. && x->imag == 0.)
        *r = *x;
    else {
        s = sqrt((fabs(x->real) + hypot(x->real,x->imag))/2);
        d = x->imag/(2*s);
        if (x->real > 0.) {
            r->real = s;
            r->imag = d;
        }
        else if (x->imag >= 0.) {
            r->real = d;
            r->imag = s;
        }
        else {
            r->real = -d;
            r->imag = -s;
        }
    }
    return;
}

static void
nc_log(npy_cdouble *x, npy_cdouble *r)
{
    double l = hypot(x->real,x->imag);
    r->imag = atan2(x->imag, x->real);
    r->real = log(l);
    return;
}

static void
nc_log1p(npy_cdouble *x, npy_cdouble *r)
{
    double l = hypot(x->real + 1.0,x->imag);
    r->imag = atan2(x->imag, x->real + 1.0);
    r->real = log(l);
    return;
}

static void
nc_exp(npy_cdouble *x, npy_cdouble *r)
{
    double a = exp(x->real);
    r->real = a*cos(x->imag);
    r->imag = a*sin(x->imag);
    return;
}

static void
nc_expm1(npy_cdouble *x, npy_cdouble *r)
{
    double a = exp(x->real);
    r->real = a*cos(x->imag) - 1.0;
    r->imag = a*sin(x->imag);
    return;
}

static void
nc_pow(npy_cdouble *a, npy_cdouble *b, npy_cdouble *r)
{
    npy_intp n;
    double ar=a->real, br=b->real, ai=a->imag, bi=b->imag;

    if (br == 0. && bi == 0.) {
        r->real = 1.;
        r->imag = 0.;
        return;
    }
    if (ar == 0. && ai == 0.) {
        r->real = 0.;
        r->imag = 0.;
        return;
    }
    if (bi == 0 && (n=(npy_intp)br) == br) {
        if (n > -100 && n < 100) {
        npy_cdouble p, aa;
        npy_intp mask = 1;
        if (n < 0) n = -n;
        aa = nc_1;
        p.real = ar; p.imag = ai;
        while (1) {
            if (n & mask)
                nc_prod(&aa,&p,&aa);
            mask <<= 1;
            if (n < mask || mask <= 0) break;
            nc_prod(&p,&p,&p);
        }
        r->real = aa.real; r->imag = aa.imag;
        if (br < 0) nc_quot(&nc_1, r, r);
        return;
        }
    }
    /* complexobject.c uses an inline version of this formula
       investigate whether this had better performance or accuracy */
    nc_log(a, r);
    nc_prod(r, b, r);
    nc_exp(r, r);
    return;
}


static void
nc_prodi(npy_cdouble *x, npy_cdouble *r)
{
    double xr = x->real;
    r->real = -x->imag;
    r->imag = xr;
    return;
}


static void
nc_acos(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble a, *pa=&a;

    nc_assign(x, pa);
    nc_prod(x,x,r);
    nc_diff(&nc_1, r, r);
    nc_sqrt(r, r);
    nc_prodi(r, r);
    nc_sum(pa, r, r);
    nc_log(r, r);
    nc_prodi(r, r);
    nc_neg(r, r);
    return;
    /* return nc_neg(nc_prodi(nc_log(nc_sum(x,nc_prod(nc_i,
       nc_sqrt(nc_diff(nc_1,nc_prod(x,x))))))));
    */
}

static void
nc_acosh(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble t, a, *pa=&a;

    nc_assign(x, pa);
    nc_sum(x, &nc_1, &t);
    nc_sqrt(&t, &t);
    nc_diff(x, &nc_1, r);
    nc_sqrt(r, r);
    nc_prod(&t, r, r);
    nc_sum(pa, r, r);
    nc_log(r, r);
    return;
    /*
      return nc_log(nc_sum(x,
      nc_prod(nc_sqrt(nc_sum(x,nc_1)), nc_sqrt(nc_diff(x,nc_1)))));
    */
}

static void
nc_asin(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble a, *pa=&a;
    nc_prodi(x, pa);
    nc_prod(x, x, r);
    nc_diff(&nc_1, r, r);
    nc_sqrt(r, r);
    nc_sum(pa, r, r);
    nc_log(r, r);
    nc_prodi(r, r);
    nc_neg(r, r);
    return;
    /*
      return nc_neg(nc_prodi(nc_log(nc_sum(nc_prod(nc_i,x),
      nc_sqrt(nc_diff(nc_1,nc_prod(x,x)))))));
    */
}


static void
nc_asinh(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble a, *pa=&a;
    nc_assign(x, pa);
    nc_prod(x, x, r);
    nc_sum(&nc_1, r, r);
    nc_sqrt(r, r);
    nc_sum(r, pa, r);
    nc_log(r, r);
    return;
    /*
      return nc_log(nc_sum(nc_sqrt(nc_sum(nc_1,nc_prod(x,x))),x));
    */
}

static void
nc_atan(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble a, *pa=&a;
    nc_diff(&nc_i, x, pa);
    nc_sum(&nc_i, x, r);
    nc_quot(r, pa, r);
    nc_log(r,r);
    nc_prod(&nc_i2, r, r);
    return;
    /*
      return nc_prod(nc_i2,nc_log(nc_quot(nc_sum(nc_i,x),nc_diff(nc_i,x))));
    */
}

static void
nc_atanh(npy_cdouble *x, npy_cdouble *r)
{
    npy_cdouble a, b, *pa=&a, *pb=&b;
    nc_assign(x, pa);
    nc_diff(&nc_1, pa, r);
    nc_sum(&nc_1, pa, pb);
    nc_quot(pb, r, r);
    nc_log(r, r);
    nc_prod(&nc_half, r, r);
    return;
    /*
      return nc_prod(nc_half,nc_log(nc_quot(nc_sum(nc_1,x),nc_diff(nc_1,x))));
    */
}

static void
nc_cos(npy_cdouble *x, npy_cdouble *r)
{
    double xr=x->real, xi=x->imag;
    r->real = cos(xr)*cosh(xi);
    r->imag = -sin(xr)*sinh(xi);
    return;
}

static void
nc_cosh(npy_cdouble *x, npy_cdouble *r)
{
    double xr=x->real, xi=x->imag;
    r->real = cos(xi)*cosh(xr);
    r->imag = sin(xi)*sinh(xr);
    return;
}


#define M_LOG10_E 0.434294481903251827651128918916605082294397

static void
nc_log10(npy_cdouble *x, npy_cdouble *r)
{
    nc_log(x, r);
    r->real *= M_LOG10_E;
    r->imag *= M_LOG10_E;
    return;
}

static void
nc_sin(npy_cdouble *x, npy_cdouble *r)
{
    double xr=x->real, xi=x->imag;
    r->real = sin(xr)*cosh(xi);
    r->imag = cos(xr)*sinh(xi);
    return;
}

static void
nc_sinh(npy_cdouble *x, npy_cdouble *r)
{
    double xr=x->real, xi=x->imag;
    r->real = cos(xi)*sinh(xr);
    r->imag = sin(xi)*cosh(xr);
    return;
}

static void
nc_tan(npy_cdouble *x, npy_cdouble *r)
{
    double sr,cr,shi,chi;
    double rs,is,rc,ic;
    double d;
    double xr=x->real, xi=x->imag;
    sr = sin(xr);
    cr = cos(xr);
    shi = sinh(xi);
    chi = cosh(xi);
    rs = sr*chi;
    is = cr*shi;
    rc = cr*chi;
    ic = -sr*shi;
    d = rc*rc + ic*ic;
    r->real = (rs*rc+is*ic)/d;
    r->imag = (is*rc-rs*ic)/d;
    return;
}

static void
nc_tanh(npy_cdouble *x, npy_cdouble *r)
{
    double si,ci,shr,chr;
    double rs,is,rc,ic;
    double d;
    double xr=x->real, xi=x->imag;
    si = sin(xi);
    ci = cos(xi);
    shr = sinh(xr);
    chr = cosh(xr);
    rs = ci*shr;
    is = si*chr;
    rc = ci*chr;
    ic = si*shr;
    d = rc*rc + ic*ic;
    r->real = (rs*rc+is*ic)/d;
    r->imag = (is*rc-rs*ic)/d;
    return;
}

static void
nc_abs(npy_cdouble *x, npy_cdouble *r)
{
    r->real = sqrt(x->real*x->real + x->imag*x->imag);
    r->imag = 0;
}

#endif // NUMEXPR_COMPLEX_FUNCTIONS_HPP
