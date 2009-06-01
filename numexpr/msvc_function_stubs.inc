/* Declare stub functions for MSVC.  It turns out that single precision
   definitions in <math.h> are actually #define'd and are not usable
   as function pointers :-/ */

inline float sqrtf2(float x) {
    return sqrtf(x);
}

inline float sinf2(float x) {
    return sinf(x);
}

inline float cosf2(float x) {
    return cosf(x);
}

inline float tanf2(float x) {
    return tanf(x);
}

inline float asinf2(float x) {
    return asinf(x);
}

inline float acosf2(float x) {
    return acosf(x);
}

inline float atanf2(float x) {
    return atanf(x);
}

inline float sinhf2(float x) {
    return sinhf(x);
}

inline float coshf2(float x) {
    return coshf(x);
}

inline float tanhf2(float x) {
    return tanhf(x);
}

inline float asinhf2(float x) {
    return asinhf(x);
}

inline float acoshf2(float x) {
    return acoshf(x);
}

inline float atanhf2(float x) {
    return atanhf(x);
}

inline float logf2(float x) {
    return logf(x);
}

inline float log1pf2(float x) {
    return log1pf(x);
}

inline float log10f2(float x) {
    return log10f(x);
}

inline float expf2(float x) {
    return expf(x);
}

inline float expm1f2(float x) {
    return expm1f(x);
}

inline float fmodf2(float x, float y) {
    return fmodf(x, y);
}

inline float atan2f2(float x, float y) {
    return atan2f(x, y);
}
