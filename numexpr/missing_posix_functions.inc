/* These functions are not included in some non-POSIX compilers,
  like MSVC 7.1  */


/* Double precision versions */

inline static double log1p(double x)
{
    double u = 1.0 + x;
    if (u == 1.0) {
        return x;
    } else {
        return log(u) * x / (u-1.0);
    }
}

inline static double expm1(double x)
{
    double u = exp(x);
    if (u == 1.0) {
        return x;
    } else if (u-1.0 == -1.0) {
        return -1;
    } else {
        return (u-1.0) * x/log(u);
    }
}

inline static double asinh(double xx)
{
    double x, d;
    int sign;
    if (xx < 0.0) {
        sign = -1;
        x = -xx;
    }
    else {
        sign = 1;
        x = xx;
    }
    if (x > 1e8) {
        d = x;
    } else {
        d = sqrt(x*x + 1.0);
    }
    return sign*log1p(x*(1.0 + x/(d+1.0)));
}

inline static double acosh(double x)
{
    return 2*log(sqrt((x+1.0)/2)+sqrt((x-1.0)/2));
}

inline static double atanh(double x)
{
    /* This definition is different from that in NumPy 1.3 and follows
    the convention of MatLab.  This will allow for double checking both
    approaches. */
    return 0.5*log((1.0+x)/(1.0-x));
}


/* Single precision versions */

inline static float log1pf(float x)
{
    return (float) log1p((double)x);
}

inline static float expm1f(float x)
{
    return (float) expm1((double)x);
}

inline static float asinhf(float x)
{
    return (float) asinh((double)x);
}

inline static float acoshf(float x)
{
    return (float) acosh((double)x);
}

inline static float atanhf(float x)
{
    return (float) atanh((double)x);
}
