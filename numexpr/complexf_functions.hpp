#ifndef NUMEXPR_COMPLEXF_FUNCTIONS_HPP
#define NUMEXPR_COMPLEXF_FUNCTIONS_HPP

/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// TODO: Could just use std::complex<float> and std::complex<float>

/* constants */
static npy_cfloat nx_1 = {1., 0.};
static npy_cfloat nx_half = {0.5, 0.};
static npy_cfloat nx_i = {0., 1.};
static npy_cfloat nx_i2 = {0., 0.5};
/*
static npy_cfloat nxf_mi = {0., -1.};
static npy_cfloat nxf_pi2 = {M_PI/2., 0.};
*/

/* *************************** WARNING *****************************
Due to the way Numexpr places the results of operations, the *x and *r
pointers do point to the same address (apparently this doesn't happen
in NumPy).  So, measures should be taken so as to not to reuse *x
after the first *r has been overwritten.
*********************************************************************
*/

static void
nx_assign(npy_cfloat *x, npy_cfloat *r)
{
  r->real = x->real;
  r->imag = x->imag;
  return;
}

static void
nx_sum(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r)
{
    r->real = a->real + b->real;
    r->imag = a->imag + b->imag;
    return;
}

static void
nx_diff(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r)
{
    r->real = a->real - b->real;
    r->imag = a->imag - b->imag;
    return;
}

static void
nx_neg(npy_cfloat *a, npy_cfloat *r)
{
    r->real = -a->real;
    r->imag = -a->imag;
    return;
}

static void
nx_conj(npy_cfloat *a, npy_cfloat *r)
{
    r->real = a->real;
    r->imag = -a->imag;
    return;
}

// RAM: What this means is if the user calls conj(x) on a float or double do a 
// null-op so the interpreter doesn't break.
// Needed for allowing the internal casting in numexpr machinery for
// conjugate operations
inline float fconjf(float x)
{
    return x;
}



static void
nx_prod(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r)
{
    float ar=a->real, br=b->real, ai=a->imag, bi=b->imag;
    r->real = ar*br - ai*bi;
    r->imag = ar*bi + ai*br;
    return;
}

static void
nx_quot(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r)
{
    float ar=a->real, br=b->real, ai=a->imag, bi=b->imag;
    float d = br*br + bi*bi;
    r->real = (ar*br + ai*bi)/d;
    r->imag = (ai*br - ar*bi)/d;
    return;
}

static void
nx_sqrt(npy_cfloat *x, npy_cfloat *r)
{
    float s,d;
    if (x->real == 0. && x->imag == 0.)
        *r = *x;
    else {
        s = sqrtf((fabsf(x->real) + hypotf(x->real,x->imag))/2);
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
nx_log(npy_cfloat *x, npy_cfloat *r)
{
    float l = hypotf(x->real,x->imag);
    r->imag = atan2f(x->imag, x->real);
    r->real = logf(l);
    return;
}

static void
nx_log1p(npy_cfloat *x, npy_cfloat *r)
{
    float l = hypotf(x->real + 1.0,x->imag);
    r->imag = atan2f(x->imag, x->real + 1.0);
    r->real = logf(l);
    return;
}

static void
nx_exp(npy_cfloat *x, npy_cfloat *r)
{
    float a = expf(x->real);
    r->real = a*cosf(x->imag);
    r->imag = a*sinf(x->imag);
    return;
}

static void
nx_expm1(npy_cfloat *x, npy_cfloat *r)
{
    float a = expf(x->real);
    r->real = a*cosf(x->imag) - 1.0;
    r->imag = a*sinf(x->imag);
    return;
}

static void
nx_pow(npy_cfloat *a, npy_cfloat *b, npy_cfloat *r)
{
    npy_intp n;
    float ar=a->real, br=b->real, ai=a->imag, bi=b->imag;

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
        npy_cfloat p, aa;
        npy_intp mask = 1;
        if (n < 0) n = -n;
        aa = nx_1;
        p.real = ar; p.imag = ai;
        while (1) {
            if (n & mask)
                nx_prod(&aa,&p,&aa);
            mask <<= 1;
            if (n < mask || mask <= 0) break;
            nx_prod(&p,&p,&p);
        }
        r->real = aa.real; r->imag = aa.imag;
        if (br < 0) nx_quot(&nx_1, r, r);
        return;
        }
    }
    /* complexobject.c uses an inline version of this formula
       investigate whether this had better performance or accuracy */
    nx_log(a, r);
    nx_prod(r, b, r);
    nx_exp(r, r);
    return;
}


static void
nx_prodi(npy_cfloat *x, npy_cfloat *r)
{
    float xr = x->real;
    r->real = -x->imag;
    r->imag = xr;
    return;
}


static void
nx_acos(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat a, *pa=&a;

    nx_assign(x, pa);
    nx_prod(x,x,r);
    nx_diff(&nx_1, r, r);
    nx_sqrt(r, r);
    nx_prodi(r, r);
    nx_sum(pa, r, r);
    nx_log(r, r);
    nx_prodi(r, r);
    nx_neg(r, r);
    return;
    /* return ncf_neg(ncf_prodi(ncf_log(ncf_sum(x,ncf_prod(ncf_i,
       ncf_sqrt(ncf_diff(ncf_1,ncf_prod(x,x))))))));
    */
}

static void
nx_acosh(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat t, a, *pa=&a;

    nx_assign(x, pa);
    nx_sum(x, &nx_1, &t);
    nx_sqrt(&t, &t);
    nx_diff(x, &nx_1, r);
    nx_sqrt(r, r);
    nx_prod(&t, r, r);
    nx_sum(pa, r, r);
    nx_log(r, r);
    return;
    /*
      return ncf_log(ncf_sum(x,
      ncf_prod(ncf_sqrt(ncf_sum(x,ncf_1)), ncf_sqrt(ncf_diff(x,ncf_1)))));
    */
}

static void
nx_asin(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat a, *pa=&a;
    nx_prodi(x, pa);
    nx_prod(x, x, r);
    nx_diff(&nx_1, r, r);
    nx_sqrt(r, r);
    nx_sum(pa, r, r);
    nx_log(r, r);
    nx_prodi(r, r);
    nx_neg(r, r);
    return;
    /*
      return ncf_neg(ncf_prodi(ncf_log(ncf_sum(ncf_prod(ncf_i,x),
      ncf_sqrt(ncf_diff(ncf_1,ncf_prod(x,x)))))));
    */
}


static void
nx_asinh(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat a, *pa=&a;
    nx_assign(x, pa);
    nx_prod(x, x, r);
    nx_sum(&nx_1, r, r);
    nx_sqrt(r, r);
    nx_sum(r, pa, r);
    nx_log(r, r);
    return;
    /*
      return ncf_log(ncf_sum(ncf_sqrt(ncf_sum(ncf_1,ncf_prod(x,x))),x));
    */
}

static void
nx_atan(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat a, *pa=&a;
    nx_diff(&nx_i, x, pa);
    nx_sum(&nx_i, x, r);
    nx_quot(r, pa, r);
    nx_log(r,r);
    nx_prod(&nx_i2, r, r);
    return;
    /*
      return ncf_prod(ncf_i2,ncf_log(ncf_quot(ncf_sum(ncf_i,x),ncf_diff(ncf_i,x))));
    */
}

static void
nx_atanh(npy_cfloat *x, npy_cfloat *r)
{
    npy_cfloat a, b, *pa=&a, *pb=&b;
    nx_assign(x, pa);
    nx_diff(&nx_1, pa, r);
    nx_sum(&nx_1, pa, pb);
    nx_quot(pb, r, r);
    nx_log(r, r);
    nx_prod(&nx_half, r, r);
    return;
    /*
      return ncf_prod(ncf_half,ncf_log(ncf_quot(ncf_sum(ncf_1,x),ncf_diff(ncf_1,x))));
    */
}

static void
nx_cos(npy_cfloat *x, npy_cfloat *r)
{
    float xr=x->real, xi=x->imag;
    r->real = cosf(xr)*coshf(xi);
    r->imag = -sinf(xr)*sinhf(xi);
    return;
}

static void
nx_cosh(npy_cfloat *x, npy_cfloat *r)
{
    float xr=x->real, xi=x->imag;
    r->real = cosf(xi)*coshf(xr);
    r->imag = sinf(xi)*sinhf(xr);
    return;
}


#define M_LOG10_E 0.434294481903251827651128918916605082294397

static void
nx_log10(npy_cfloat *x, npy_cfloat *r)
{
    nx_log(x, r);
    r->real *= M_LOG10_E;
    r->imag *= M_LOG10_E;
    return;
}

static void
nx_sin(npy_cfloat *x, npy_cfloat *r)
{
    float xr=x->real, xi=x->imag;
    r->real = sinf(xr)*coshf(xi);
    r->imag = cosf(xr)*sinhf(xi);
    return;
}

static void
nx_sinh(npy_cfloat *x, npy_cfloat *r)
{
    float xr=x->real, xi=x->imag;
    r->real = cosf(xi)*sinhf(xr);
    r->imag = sinf(xi)*coshf(xr);
    return;
}

static void
nx_tan(npy_cfloat *x, npy_cfloat *r)
{
    float sr,cr,shi,chi;
    float rs,is,rc,ic;
    float d;
    float xr=x->real, xi=x->imag;
    sr = sinf(xr);
    cr = cosf(xr);
    shi = sinhf(xi);
    chi = coshf(xi);
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
nx_tanh(npy_cfloat *x, npy_cfloat *r)
{
    float si,ci,shr,chr;
    float rs,is,rc,ic;
    float d;
    float xr=x->real, xi=x->imag;
    si = sinf(xi);
    ci = cosf(xi);
    shr = sinhf(xr);
    chr = coshf(xr);
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
nx_abs(npy_cfloat *x, npy_cfloat *r)
{
    r->real = sqrtf(x->real*x->real + x->imag*x->imag);
    r->imag = 0;
}

#endif // NUMEXPR_COMPLEXF_FUNCTIONS_HPP
