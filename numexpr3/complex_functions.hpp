#ifndef NUMEXPR_COMPLEX_FUNCTIONS_HPP
#define NUMEXPR_COMPLEX_FUNCTIONS_HPP

/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

// constants
static npy_complex64 Z_1 = {1., 0.};
static npy_complex64 Z_HALF = {0.5, 0.};
static npy_complex64 Z_J = {0., 1.};
static npy_complex64 Z_J2 = {0., 0.5};
                             
static npy_complex128 C_1 = {1., 0.};
static npy_complex128 C_HALF = {0.5, 0.};
static npy_complex128 C_J = {0., 1.};
static npy_complex128 C_J2 = {0., 0.5};
                             
/* In NumpPy complex is not the C99 complex but a struct:

#if NPY_SIZEOF_COMPLEX_DOUBLE != 2 * NPY_SIZEOF_DOUBLE
#error npy_cdouble definition is not compatible with C99 complex definition ! \
        Please contact NumPy maintainers and give detailed information about your \
        compiler and platform
#endif
typedef struct { double real, imag; } npy_cdouble;
*/

                        
// TODO: apparently the inline _should_ be returning the value instead of 
// pushing to &r.  Make a backup before you try it.   
// http://www.gamasutra.com/view/feature/4248/designing_fast_crossplatform_simd_.php
// He also advises _not_ overloading functions. It's probably _not_ 
// the case for the Intel compiler, but for modern gcc 5.4?  


// abs() has been changed to return npy_float and npy_double, like NumPy

static void
nc_real( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_float32 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I].real;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1].real;
        }
    }
        
}

static void
nc_real( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_float64 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I].real;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1].real;
        }
    }
}
    
static void
nc_imag( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_float32 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1].imag;
        }
    }
}

static void
nc_imag( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_float64 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1].imag;
        }
    }
}
    
static void
nc_complex( npy_intp n, npy_float32 *a, npy_intp sb1, npy_float32 *b, npy_intp sb2, npy_complex64 *r )
{
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I];
            r[I].imag = b[I];
        }
    }
    else {
        sb1 /= sizeof(npy_float32);
        sb2 /= sizeof(npy_float32);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I*sb1];
            r[I].imag = b[I*sb2];
        }
    }
}
  
static void
nc_complex( npy_intp n, npy_float64 *a, npy_intp sb1, npy_float64 *b, npy_intp sb2, npy_complex128 *r )
{
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I];
            r[I].imag = b[I];
        }
    }
    else {
        sb1 /= sizeof(npy_float64);
        sb2 /= sizeof(npy_float64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I*sb1];
            r[I].imag = b[I*sb2];
        }
    }
}
    
static inline void
_inline_add( npy_complex64 a, npy_complex64 b, npy_complex64 &r )
{
    r.real = a.real + b.real;
    r.imag = a.imag + b.imag;
}

static inline void
_inline_add( npy_complex128 a, npy_complex128 b, npy_complex128 &r )
{
    r.real = a.real + b.real;
    r.imag = a.imag + b.imag;
}

static void
nc_add( npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r )
{
    if( sb1 == sizeof(npy_complex64) && sb2 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_add( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex64);
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_add( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}
    
static void
nc_add( npy_intp n, npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r )
{
    if( sb1 == sizeof(npy_complex128) && sb2 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_add( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex128);
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_add( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}
    
static inline void
_inline_sub( npy_complex64 a, npy_complex64 b, npy_complex64 &r )
{
    r.real = a.real - b.real;
    r.imag = a.imag - b.imag;
}

static inline void
_inline_sub( npy_complex128 a, npy_complex128 b, npy_complex128 &r  )
{
    r.real = a.real - b.real;
    r.imag = a.imag - b.imag;
}

static void
nc_sub( npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r )
{
    if( sb1 == sizeof(npy_complex64) && sb2 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex64);
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

static void
nc_sub( npy_intp n,npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) && sb2 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex128);
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

static inline void
_inline_neg( npy_complex64 a, npy_complex64 &r )
{
    r.real = -a.real;
    r.imag = -a.imag;
}

static inline void
_inline_neg( npy_complex128 a, npy_complex128 &r )
{
    r.real = -a.real;
    r.imag = -a.imag;
}

static void
nc_neg( npy_intp n,npy_complex64 *a, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_neg( a[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_neg( a[I*sb1], r[I] );
        }
    }
}

static void
nc_neg( npy_intp n,npy_complex128 *a, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_neg( a[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_neg( a[I*sb1], r[I] );
        }
    }
}

static void
nc_conj(npy_intp n,npy_complex64 *a, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I].real;
            r[I].imag = -a[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I*sb1].real;
            r[I].imag = -a[I*sb1].imag;
        }
    }
}

static void
nc_conj(npy_intp n,npy_complex128 *a, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I].real;
            r[I].imag = -a[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = a[I*sb1].real;
            r[I].imag = -a[I*sb1].imag;
        }
    }
}
    

    
// Needed for allowing the internal casting in numexpr machinery for
// conjugate operations
// RAM: What this comment means is if the user calls conj(x) on a npy_float32 or 
// npy_float64 do a null-op so the interpreter doesn't break.
static void
fconj( npy_intp n, npy_float32 *x, npy_intp sb1, npy_float32 *r )
{
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I];
        }
    }
    else {
        sb1 /= sizeof(npy_float32);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1];
        }        
    }
}

static void
fconj( npy_intp n, npy_float64 *x, npy_intp sb1, npy_float64 *r )
{
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I];
        }
    }
    else {
        sb1 /= sizeof(npy_float64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = x[I*sb1];
        }        
    }
}

static inline void
_inline_mul( npy_complex64 a, npy_complex64 b, npy_complex64 &r)
{
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
}

static inline void
_inline_mul( npy_complex128 a, npy_complex128 b, npy_complex128 &r)
{
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
}
//_return_ variants are for use inside ternary operator
static inline npy_complex64
_return_mul( npy_complex64 a, npy_complex64 b )
{
    npy_complex64 r;
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
    return r;
}
//_return_ variants are for use inside ternary operator
static inline npy_complex128
_return_mul( npy_complex128 a, npy_complex128 b )
{
    npy_complex128 r;
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
    return r;
}

static void
nc_mul( npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) && sb2 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_mul( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex64);
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_mul( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

static void
nc_mul( npy_intp n, npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) && sb2 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_mul( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex128);
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_mul( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

// The naive complex division function here is not very accurate and 
// sometimes fails the autotest assert.
// Probably there's something better...
// https://arxiv.org/pdf/1210.4539.pdf
//  or
// https://arxiv.org/pdf/1608.07596.pdf
// Most complex division algorithms seem to have a branch however, which
// would take a bit of work to avoid.
static inline void
_inline_div( npy_complex64 a, npy_complex64 b, npy_complex64 &r)
{
    npy_float32 d = 1.0f/(b.real*b.real + b.imag*b.imag);
    r.real = (a.real*b.real + a.imag*b.imag)*d;
    r.imag = (a.imag*b.real - a.real*b.imag)*d;
 
}
static inline void
_inline_div( npy_complex128 a, npy_complex128 b, npy_complex128 &r)
{
    npy_float64 d = 1.0/(b.real*b.real + b.imag*b.imag);
    r.real = (a.real*b.real + a.imag*b.imag)*d;
    r.imag = (a.imag*b.real - a.real*b.imag)*d;
}

static inline npy_complex64
_return_div( npy_complex64 a, npy_complex64 b )
{
    npy_complex64 r;
    npy_float32 d = 1.0f / (b.real*b.real + b.imag*b.imag);
    r.real = (a.real*b.real + a.imag*b.imag)*d;
    r.imag = (a.imag*b.real - a.real*b.imag)*d;
    return r;
}

static inline npy_complex128
_return_div( npy_complex128 a, npy_complex128 b )
{
    npy_complex128 r;
    npy_float64 d = 1.0 / (b.real*b.real + b.imag*b.imag);
    r.real = (a.real*b.real + a.imag*b.imag)*d;
    r.imag = (a.imag*b.real - a.real*b.imag)*d;
    return r;
}

static void
nc_div( npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) && sb2 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_div( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex64);
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_div( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

static void
nc_div( npy_intp n, npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) && sb2 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_div( a[I], b[I], r[I] );
        }
    }
    else {
        sb2 /= sizeof(npy_complex128);
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_div( a[I*sb1], b[I*sb2], r[I] );
        }
    }
}

// RAM: this function branches a lot. Search for a branchless sqrt? 
static inline void
_inline_sqrt( npy_complex64 x, npy_complex64 &r ) 
{
    npy_float32 s, d;
    
    if (x.real == 0. && x.imag == 0.)
        r = x;
    else {
        s = sqrtf((fabsf(x.real) + hypotf(x.real,x.imag))/2);
        d = x.imag/(2*s);
        if (x.real > 0.) {
            r.real = s;
            r.imag = d;
        }
        else if (x.imag >= 0.) {
            r.real = d;
            r.imag = s;
        }
        else {
            r.real = -d;
            r.imag = -s;
        }
    }
}
    
static inline void
_inline_sqrt( npy_complex128 x, npy_complex128 &r ) 
{
    npy_float64 s, d;
    
    if (x.real == 0. && x.imag == 0.)
        r = x;
    else {
        s = sqrt((fabs(x.real) + hypot(x.real,x.imag))/2);
        d = x.imag/(2*s);
        if (x.real > 0.) {
            r.real = s;
            r.imag = d;
        }
        else if (x.imag >= 0.) {
            r.real = d;
            r.imag = s;
        }
        else {
            r.real = -d;
            r.imag = -s;
        }
    }
}

static void
nc_sqrt( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sqrt( x[I], r[I] ); 
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sqrt( x[I*sb1], r[I] ); 
        }
    }
}
        
static void
nc_sqrt( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sqrt( x[I], r[I] ); 
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sqrt( x[I*sb1], r[I] ); 
        }
    }
}


static void
nc_abs( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_float32 *r )
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = sqrtf( x[I].real*x[I].real + x[I].imag*x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = sqrtf( x[I*sb1].real*x[I*sb1].real + x[I*sb1].imag*x[I*sb1].imag);
        }
    }
}

static void
nc_abs( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_float64 *r )
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = sqrt( x[I].real*x[I].real + x[I].imag*x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = sqrt( x[I*sb1].real*x[I*sb1].real + x[I*sb1].imag*x[I*sb1].imag);
        }
    }
}
    
static void
nc_abs2( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_float32 *r )
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = x[I].real*x[I].real + x[I].imag*x[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = x[I*sb1].real*x[I*sb1].real + x[I*sb1].imag*x[I*sb1].imag;
        }
    }
}

static void
nc_abs2( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_float64 *r )
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = x[I].real*x[I].real + x[I].imag*x[I].imag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
             r[I] = x[I*sb1].real*x[I*sb1].real + x[I*sb1].imag*x[I*sb1].imag;
        }
    }
}
    
static inline void
_inline_log( npy_complex64 x, npy_complex64 &r )
{
    r.imag = atan2f(x.imag, x.real);
    r.real = logf( hypotf(x.real,x.imag) );
}

static inline void
_inline_log( npy_complex128 x, npy_complex128 &r )
{
    r.imag = atan2(x.imag, x.real);
    r.real = log( hypot(x.real,x.imag) );
}

static void
nc_log( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I], r[I] ); 
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I*sb1], r[I] ); 
        }
    }
}

static void
nc_log( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I], r[I] ); 
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I*sb1], r[I] ); 
        }
    }
}

static void
nc_log1p( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].imag = atan2f(x[I].imag, x[I].real + 1.0f);
            r[I].real = logf( hypotf(x[I].real + 1.0f,x[I].imag) );
        }
    } 
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].imag = atan2f(x[I*sb1].imag, x[I*sb1].real + 1.0f);
            r[I].real = logf( hypotf(x[I*sb1].real + 1.0f,x[I*sb1].imag) );
        }
    }
}

static void
nc_log1p( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].imag = atan2(x[I].imag, x[I].real + 1.0);
            r[I].real = log( hypot(x[I].real + 1.0,x[I].imag) );
        }
    } 
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].imag = atan2(x[I*sb1].imag, x[I*sb1].real + 1.0);
            r[I].real = log( hypot(x[I*sb1].real + 1.0,x[I*sb1].imag) );
        }
    }
}

static inline void
_inline_exp( npy_complex64 x, npy_complex64 &r )
{
    npy_float32 a = exp(x.real);
    r.real = a*cosf(x.imag);
    r.imag = a*sinf(x.imag);
}

static inline void
_inline_exp( npy_complex128 x, npy_complex128 &r )
{
    npy_float64 a = exp(x.real);
    r.real = a*cos(x.imag);
    r.imag = a*sin(x.imag);
}

static void
nc_exp( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_exp( x[I], r[I] );
        }
    } 
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_exp( x[I*sb1], r[I] );
        }
    }
}


static void
nc_exp( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_exp( x[I], r[I] );
        }
    } 
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_exp( x[I*sb1], r[I] );
        }
    }
}

static void
nc_expm1( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_float32 a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = expf(x[I].real);
            r[I].real = a*cosf(x[I].imag) - 1.0f;
            r[I].imag = a*sinf(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);    
        for( npy_intp I = 0; I < n; I++ ) {
            a = expf(x[I*sb1].real);
            r[I].real = a*cosf(x[I*sb1].imag) - 1.0f;
            r[I].imag = a*sinf(x[I*sb1].imag);
        }     
    }
}

static void
nc_expm1( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_float64 a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = exp(x[I].real);
            r[I].real = a*cos(x[I].imag) - 1.0;
            r[I].imag = a*sin(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);    
        for( npy_intp I = 0; I < n; I++ ) {
            a = exp(x[I*sb1].real);
            r[I].real = a*cos(x[I*sb1].imag) - 1.0;
            r[I].imag = a*sin(x[I*sb1].imag);
        }     
    }
}



// RAM: algorithm branches, and is really not designed for a modern CPU...
// TODO: we should be able to branch the exp>-100 && exp<100 test before the 
// for loop.
static void
nc_pow( npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r)
{
    npy_intp exp, mask;
    npy_complex64 p, aa;
    sb1 /= sizeof(npy_complex64);
    sb2 /= sizeof(npy_complex64);
                 
    for( int I = 0; I < n; I++ ) {
        if (b[I*sb2].real == 0. && b[I*sb2].imag == 0.) {
            r[I].real = 1.;
            r[I].imag = 0.;
            continue;
        }
        if (a[I*sb1].real == 0. && a[I*sb1].imag == 0.) {
            r[I].real = 0.;
            r[I].imag = 0.;
            continue;
        }
        if (b[I*sb2].imag == 0 && (exp=(npy_intp)b[I*sb2].real) == b[I*sb2].real) {
            if (exp > -100 && exp < 100) {
                mask = 1;
                exp = (exp < 0) ? -exp : exp;
                      
                aa = Z_1;
                p.real = a[I*sb1].real; p.imag = a[I*sb1].imag;
                while (1) {
                    aa = (exp & mask) ? _return_mul( aa, p ) : aa;
                         
                    mask <<= 1;
                    if (n < mask || mask <= 0) break;
                    p = _return_mul( p, p );
                }
                r[I].real = aa.real; 
                r[I].imag = aa.imag;
                r[I] = (b[I*sb2].real < 0) ? _return_div( Z_1, r[I] ) : r[I];
                continue;
            }
        }
        
        _inline_log( a[I*sb1], r[I] );
        _inline_mul(r[I], b[I*sb2], r[I] );
        _inline_exp(r[I], r[I] );
    }
}
        
static void
nc_pow( npy_intp n, npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r)
{
    npy_intp exp, mask;
    npy_complex128 p, aa;
    sb1 /= sizeof(npy_complex128);
    sb2 /= sizeof(npy_complex128);
                 
    for( int I = 0; I < n; I++ ) {
        if (b[I*sb2].real == 0. && b[I*sb2].imag == 0.) {
            r[I].real = 1.;
            r[I].imag = 0.;
            continue;
        }
        if (a[I*sb1].real == 0. && a[I*sb1].imag == 0.) {
            r[I].real = 0.;
            r[I].imag = 0.;
            continue;
        }
        if (b[I*sb2].imag == 0 && (exp=(npy_intp)b[I*sb2].real) == b[I*sb2].real) {
            if (exp > -100 && exp < 100) {
                mask = 1;
                exp = (exp < 0) ? -exp : exp;
                      
                aa = C_1;
                p.real = a[I*sb1].real; p.imag = a[I*sb1].imag;
                while (1) {
                    aa = (exp & mask) ? _return_mul( aa, p ) : aa;
                         
                    mask <<= 1;
                    if (n < mask || mask <= 0) break;
                    p = _return_mul( p, p );
                }
                r[I].real = aa.real; 
                r[I].imag = aa.imag;
                r[I] = (b[I*sb2].real < 0) ? _return_div( C_1, r[I] ) : r[I];
                continue;
            }
        }
        
        _inline_log( a[I*sb1], r[I] );
        _inline_mul(r[I], b[I*sb2], r[I] );
        _inline_exp(r[I], r[I] );
    }
}

// RAM: these functions multiply the argument by the imaginary number, j,
// and they are always used inline.
static inline void
_inline_muli( npy_complex64 x, npy_complex64 &r)
{
    r.real = -x.imag;
    r.imag = x.real;
}

static inline void
_inline_muli( npy_complex128 x, npy_complex128 &r)
{
    r.real = -x.imag;
    r.imag = x.real;
}

static void
nc_acos( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_mul( x[I], x[I], r[I] );
            _inline_sub( Z_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I] , r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I]);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_mul( x[I*sb1], x[I*sb1], r[I] );
            _inline_sub( Z_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I] , r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I]);
        }  
    }
}

static void
nc_acos( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_mul( x[I], x[I], r[I] );
            _inline_sub( C_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I] , r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I]);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_mul( x[I*sb1], x[I*sb1], r[I] );
            _inline_sub( C_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I] , r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I]);
        }  
    }
}

static void
nc_acosh( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 t, a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_add( Z_1, x[I], t );
            _inline_sqrt( t, t );
            _inline_sub( x[I], Z_1, r[I] );
            _inline_sqrt( r[I] , r[I] );
            _inline_mul( t, r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_add( Z_1, x[I*sb1], t );
            _inline_sqrt( t, t );
            _inline_sub( x[I*sb1], Z_1, r[I] );
            _inline_sqrt( r[I] , r[I] );
            _inline_mul( t, r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
        }    
    }
}

static void
nc_acosh( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 t, a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_add( C_1, x[I], t );
            _inline_sqrt( t, t );
            _inline_sub( x[I], C_1, r[I] );
            _inline_sqrt( r[I] , r[I] );
            _inline_mul( t, r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_add( C_1, x[I*sb1], t );
            _inline_sqrt( t, t );
            _inline_sub( x[I*sb1], C_1, r[I] );
            _inline_sqrt( r[I] , r[I] );
            _inline_mul( t, r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
        }    
    }
}

static void
nc_asin( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_muli( x[I], a );
            _inline_mul( x[I], x[I], r[I] );
            _inline_sub( Z_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_muli( x[I*sb1], a );
            _inline_mul( x[I*sb1], x[I*sb1], r[I] );
            _inline_sub( Z_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I] );
        }     
    }
}

static void
nc_asin( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_muli( x[I], a );
            _inline_mul( x[I], x[I], r[I] );
            _inline_sub( C_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_muli( x[I*sb1], a );
            _inline_mul( x[I*sb1], x[I*sb1], r[I] );
            _inline_sub( C_1, r[I], r[I] );
            _inline_sqrt( r[I], r[I] );
            _inline_add( a, r[I], r[I] );
            _inline_log( r[I], r[I] );
            _inline_muli( r[I], r[I] );
            _inline_neg( r[I], r[I] );
        }     
    }
}

static void
nc_asinh( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_mul( x[I], x[I], r[I]);
            _inline_add( Z_1, r[I], r[I]);
            _inline_sqrt( r[I], r[I] );
            _inline_add( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_mul( x[I*sb1], x[I*sb1], r[I]);
            _inline_add( Z_1, r[I], r[I]);
            _inline_sqrt( r[I], r[I] );
            _inline_add( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
        }        
    }
}

static void
nc_asinh( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_mul( x[I], x[I], r[I]);
            _inline_add( C_1, r[I], r[I]);
            _inline_sqrt( r[I], r[I] );
            _inline_add( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_mul( x[I*sb1], x[I*sb1], r[I]);
            _inline_add( C_1, r[I], r[I]);
            _inline_sqrt( r[I], r[I] );
            _inline_add( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
        }        
    }
}

static void
nc_atan( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 a;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( Z_J, x[I], a );
            _inline_add( Z_J, x[I], r[I] );
            _inline_div( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
            _inline_mul( Z_J2, r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( Z_J, x[I*sb1], a );
            _inline_add( Z_J, x[I*sb1], r[I] );
            _inline_div( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
            _inline_mul( Z_J2, r[I], r[I] );
        }
    }
}

static void
nc_atan( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 a;
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( C_J, x[I], a );
            _inline_add( C_J, x[I], r[I] );
            _inline_div( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
            _inline_mul( C_J2, r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_sub( C_J, x[I*sb1], a );
            _inline_add( C_J, x[I*sb1], r[I] );
            _inline_div( r[I], a, r[I] );
            _inline_log( r[I], r[I] );
            _inline_mul( C_J2, r[I], r[I] );
        }
    }
}

static void
nc_atanh( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_complex64 a, b;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_sub( Z_1, a, r[I] );
            _inline_add( Z_1, a, b );
            _inline_div( b, r[I], r[I] );
            _inline_log( r[I], r[I]);
            _inline_mul( Z_HALF, r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_sub( Z_1, a, r[I] );
            _inline_add( Z_1, a, b );
            _inline_div( b, r[I], r[I] );
            _inline_log( r[I], r[I]);
            _inline_mul( Z_HALF, r[I], r[I] );
        }
    }
}

static void
nc_atanh( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_complex128 a, b;
    
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I];
            _inline_sub( C_1, a, r[I] );
            _inline_add( C_1, a, b );
            _inline_div( b, r[I], r[I] );
            _inline_log( r[I], r[I]);
            _inline_mul( C_HALF, r[I], r[I] );
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            a = x[I*sb1];
            _inline_sub( C_1, a, r[I] );
            _inline_add( C_1, a, b );
            _inline_div( b, r[I], r[I] );
            _inline_log( r[I], r[I]);
            _inline_mul( C_HALF, r[I], r[I] );
        }
    }
}

static void
nc_cos( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I].real)*coshf(x[I].imag);
            r[I].imag = -sinf(x[I].real)*sinhf(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I*sb1].real)*coshf(x[I*sb1].imag);
            r[I].imag = -sinf(x[I*sb1].real)*sinhf(x[I*sb1].imag);
        }
    }
}

static void
nc_cos( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I].real)*cosh(x[I].imag);
            r[I].imag = -sin(x[I].real)*sinh(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I*sb1].real)*cosh(x[I*sb1].imag);
            r[I].imag = -sin(x[I*sb1].real)*sinh(x[I*sb1].imag);
        }
    }
}

static void
nc_cosh( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I].imag)*coshf(x[I].real);
            r[I].imag = sinf(x[I].imag)*sinhf(x[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);  
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I*sb1].imag)*coshf(x[I*sb1].real);
            r[I].imag = sinf(x[I*sb1].imag)*sinhf(x[I*sb1].real);
        }             
    }
}

static void
nc_cosh( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I].imag)*cosh(x[I].real);
            r[I].imag = sin(x[I].imag)*sinh(x[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);  
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I*sb1].imag)*cosh(x[I*sb1].real);
            r[I].imag = sin(x[I*sb1].imag)*sinh(x[I*sb1].real);
        }             
    }
}

#define M_LOG10_E 0.434294481903251827651128918916605082294397
#define M_LOG10_EF 0.434294481903251827651128918916605082294397f

static void
nc_log10( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I], r[I] );
            r[I].real *= M_LOG10_EF;          
            r[I].imag *= M_LOG10_EF;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I*sb1], r[I] );
            r[I].real *= M_LOG10_EF;          
            r[I].imag *= M_LOG10_EF;
        }
    }
}

static void
nc_log10( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I], r[I] );
            r[I].real *= M_LOG10_E;          
            r[I].imag *= M_LOG10_E;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            _inline_log( x[I*sb1], r[I] );
            r[I].real *= M_LOG10_E;          
            r[I].imag *= M_LOG10_E;
        }
    }
}

static void
nc_sin( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = sinf(x[I].real)*coshf(x[I].imag);
            r[I].imag = cosf(x[I].real)*sinhf(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = sinf(x[I*sb1].real)*coshf(x[I*sb1].imag);
            r[I].imag = cosf(x[I*sb1].real)*sinhf(x[I*sb1].imag);
        }
    }
}

static void
nc_sin( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = sin(x[I].real)*cosh(x[I].imag);
            r[I].imag = cos(x[I].real)*sinh(x[I].imag);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = sin(x[I*sb1].real)*cosh(x[I*sb1].imag);
            r[I].imag = cos(x[I*sb1].real)*sinh(x[I*sb1].imag);
        }
    }
}

static void
nc_sinh( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I].imag)*sinhf(x[I].real);
            r[I].imag = sinf(x[I].imag)*coshf(x[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cosf(x[I].imag)*sinhf(x[I].real);
            r[I].imag = sinf(x[I].imag)*coshf(x[I].real);
        }
    }
}

static void
nc_sinh( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I].imag)*sinh(x[I].real);
            r[I].imag = sin(x[I].imag)*cosh(x[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            r[I].real = cos(x[I].imag)*sinh(x[I].real);
            r[I].imag = sin(x[I].imag)*cosh(x[I].real);
        }
    }
}

static void
nc_tan( npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_float32 sr,cr,shi,chi;
    npy_float32 rs,is,rc,ic;
    npy_float32 d;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            sr = sinf(x[I].real);
            cr = cosf(x[I].real);
            shi = sinhf(x[I].imag);
            chi = coshf(x[I].imag);
            rs = sr*chi;
            is = cr*shi;
            rc = cr*chi;
            ic = -sr*shi;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            sr = sinf(x[I*sb1].real);
            cr = cosf(x[I*sb1].real);
            shi = sinhf(x[I*sb1].imag);
            chi = coshf(x[I*sb1].imag);
            rs = sr*chi;
            is = cr*shi;
            rc = cr*chi;
            ic = -sr*shi;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
}

static void
nc_tan( npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_float64 sr,cr,shi,chi;
    npy_float64 rs,is,rc,ic;
    npy_float64 d;
    
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            sr = sin(x[I].real);
            cr = cos(x[I].real);
            shi = sinh(x[I].imag);
            chi = cosh(x[I].imag);
            rs = sr*chi;
            is = cr*shi;
            rc = cr*chi;
            ic = -sr*shi;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            sr = sin(x[I*sb1].real);
            cr = cos(x[I*sb1].real);
            shi = sinh(x[I*sb1].imag);
            chi = cosh(x[I*sb1].imag);
            rs = sr*chi;
            is = cr*shi;
            rc = cr*chi;
            ic = -sr*shi;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
}

static void
nc_tanh(npy_intp n, npy_complex64 *x, npy_intp sb1, npy_complex64 *r)
{
    npy_float32 si,ci,shr,chr;
    npy_float32 rs,is,rc,ic;
    npy_float32 d;
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            si = sinf(x[I].imag);
            ci = cosf(x[I].imag);
            shr = sinhf(x[I].real);
            chr = coshf(x[I].real);
            rs = ci*shr;
            is = si*chr;
            rc = ci*chr;
            ic = si*shr;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            si = sinf(x[I*sb1].imag);
            ci = cosf(x[I*sb1].imag);
            shr = sinhf(x[I*sb1].real);
            chr = coshf(x[I*sb1].real);
            rs = ci*shr;
            is = si*chr;
            rc = ci*chr;
            ic = si*shr;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
}

static void
nc_tanh(npy_intp n, npy_complex128 *x, npy_intp sb1, npy_complex128 *r)
{
    npy_float64 si,ci,shr,chr;
    npy_float64 rs,is,rc,ic;
    npy_float64 d;
    
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            si = sin(x[I].imag);
            ci = cos(x[I].imag);
            shr = sinh(x[I].real);
            chr = cosh(x[I].real);
            rs = ci*shr;
            is = si*chr;
            rc = ci*chr;
            ic = si*shr;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            si = sin(x[I*sb1].imag);
            ci = cos(x[I*sb1].imag);
            shr = sinh(x[I*sb1].real);
            chr = cosh(x[I*sb1].real);
            rs = ci*shr;
            is = si*chr;
            rc = ci*chr;
            ic = si*shr;
            d = rc*rc + ic*ic;
            r[I].real = (rs*rc+is*ic)/d;
            r[I].imag = (is*rc-rs*ic)/d;
        }
    }
}

static void 
nc_angle(npy_intp n, npy_complex64 *a, npy_intp sb1, npy_float32 *r) {

    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = atan2f(a[I].imag, a[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            r[I*sb1] = atan2f(a[I*sb1].imag, a[I*sb1].real);
        }
    }
}

static void 
nc_angle(npy_intp n, npy_complex128 *a, npy_intp sb1, npy_float64 *r) {

    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            r[I] = atan2(a[I].imag, a[I].real);
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            r[I*sb1] = atan2(a[I*sb1].imag, a[I*sb1].real);
        }
    }
}


// RAM: cross-power-spectrum (for Phase Correlation).
// https://en.wikipedia.org/wiki/Phase_correlation
static void
nc_crosspower(npy_intp n, npy_complex64 *a, npy_intp sb1, npy_complex64 *b, npy_intp sb2, npy_complex64 *r) {
    npy_float32 mag;
    npy_complex64 prod; // Not-inlining this into r should improve SIMD performance.

    if( sb1 == sizeof(npy_complex64) && sb2 == sizeof(npy_complex64) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            prod.real =  a[I].real*b[I].real + a[I].imag*b[I].imag;
            prod.imag = -a[I].real*b[I].imag + a[I].imag*b[I].real;
            // Compute magnitude 
            mag = 1.0f/sqrtf(prod.real*prod.real + prod.imag*prod.imag);
            // Normalize
            r[I].real = prod.real*mag;
            r[I].imag = prod.imag*mag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex64);
        sb2 /= sizeof(npy_complex64);
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            prod.real =  a[I*sb1].real*b[I*sb2].real + a[I*sb1].imag*b[I*sb2].imag;
            prod.imag = -a[I*sb1].real*b[I*sb2].imag + a[I*sb1].imag*b[I*sb2].real;
            // Compute magnitude 
            mag = 1.0f/sqrtf(prod.real*prod.real + prod.imag*prod.imag);
            // Normalize
            r[I].real = prod.real*mag;
            r[I].imag = prod.imag*mag;
        }
    }
}

static void
nc_crosspower(npy_intp n, npy_complex128 *a, npy_intp sb1, npy_complex128 *b, npy_intp sb2, npy_complex128 *r) {
    npy_float64 mag;
    npy_complex128 prod; // Not-inlining this into r should improve SIMD performance.

    if( sb1 == sizeof(npy_complex128) && sb2 == sizeof(npy_complex128) ) { // Aligned
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            prod.real =  a[I].real*b[I].real + a[I].imag*b[I].imag;
            prod.imag = -a[I].real*b[I].imag + a[I].imag*b[I].real;
            // Compute magnitude 
            mag = 1.0/sqrt(prod.real*prod.real + prod.imag*prod.imag);
            // Normalize
            r[I].real = prod.real*mag;
            r[I].imag = prod.imag*mag;
        }
    }
    else {
        sb1 /= sizeof(npy_complex128);
        sb2 /= sizeof(npy_complex128);
        for( npy_intp I = 0; I < n; I++ ) {
            // Take product of A and complex conjugate of B
            prod.real =  a[I*sb1].real*b[I*sb2].real + a[I*sb1].imag*b[I*sb2].imag;
            prod.imag = -a[I*sb1].real*b[I*sb2].imag + a[I*sb1].imag*b[I*sb2].real;
            // Compute magnitude 
            mag = 1.0/sqrt(prod.real*prod.real + prod.imag*prod.imag);
            // Normalize
            r[I].real = prod.real*mag;
            r[I].imag = prod.imag*mag;
        }
    }
}

#endif // NUMEXPR_COMPLEX_FUNCTIONS_HPP
