// -*- c-mode -*-
/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
*********************************************************************
*/


// These #if blocks make it easier to query this file, without having
// to define every row function before #including it. 
#ifndef FUNC_F4F4
#define ELIDE_FUNC_F4F4
#define FUNC_F4F4(...)
#endif
FUNC_F4F4(FUNC_SQRT_F4F4,    "sqrt_f4f4",     sqrtf,  sqrtf2,  vsSqrt)
FUNC_F4F4(FUNC_SIN_F4F4,     "sin_f4f4",      sinf,   sinf2,   vsSin)
FUNC_F4F4(FUNC_COS_F4F4,     "cos_f4f4",      cosf,   cosf2,   vsCos)
FUNC_F4F4(FUNC_TAN_F4F4,     "tan_f4f4",      tanf,   tanf2,   vsTan)
FUNC_F4F4(FUNC_ARCSIN_F4F4,  "arcsin_f4f4",   asinf,  asinf2,  vsAsin)
FUNC_F4F4(FUNC_ARCCOS_F4F4,  "arccos_f4f4",   acosf,  acosf2,  vsAcos)
FUNC_F4F4(FUNC_ARCTAN_F4F4,  "arctan_f4f4",   atanf,  atanf2,  vsAtan)
FUNC_F4F4(FUNC_SINH_F4F4,    "sinh_f4f4",     sinhf,  sinhf2,  vsSinh)
FUNC_F4F4(FUNC_COSH_F4F4,    "cosh_f4f4",     coshf,  coshf2,  vsCosh)
FUNC_F4F4(FUNC_TANH_F4F4,    "tanh_f4f4",     tanhf,  tanhf2,  vsTanh)
FUNC_F4F4(FUNC_ARCSINH_F4F4, "arcsinh_f4f4",  asinhf, asinhf2, vsAsinh)
FUNC_F4F4(FUNC_ARCCOSH_F4F4, "arccosh_f4f4",  acoshf, acoshf2, vsAcosh)
FUNC_F4F4(FUNC_ARCTANH_F4F4, "arctanh_f4f4",  atanhf, atanhf2, vsAtanh)
FUNC_F4F4(FUNC_LOG_F4F4,     "log_f4f4",      logf,   logf2,   vsLn)
FUNC_F4F4(FUNC_LOG1P_F4F4,   "log1p_f4f4",    log1pf, log1pf2, vsLog1p)
FUNC_F4F4(FUNC_LOG10_F4F4,   "log10_f4f4",    log10f, log10f2, vsLog10)
FUNC_F4F4(FUNC_EXP_F4F4,     "exp_f4f4",      expf,   expf2,   vsExp)
FUNC_F4F4(FUNC_EXPM1_F4F4,   "expm1_f4f4",    expm1f, expm1f2, vsExpm1)
FUNC_F4F4(FUNC_ABS_F4F4,     "absolute_f4f4", fabsf,  fabsf2,  vsAbs)
FUNC_F4F4(FUNC_CONJ_F4F4,    "conjugate_f4f4",fconjf, fconjf2, vsConj)
FUNC_F4F4(FUNC_F4F4_LAST,    NULL,          NULL,   NULL,    NULL)
#ifdef ELIDE_FUNC_F4F4
#undef ELIDE_FUNC_F4F4
#undef FUNC_F4F4
#endif

#ifndef FUNC_F4F4F4
#define ELIDE_FUNC_F4F4F4
#define FUNC_F4F4F4(...)
#endif
FUNC_F4F4F4(FUNC_FMOD_F4F4F4,    "fmod_f4f4f4",    fmodf,  fmodf2,  vsfmod)
FUNC_F4F4F4(FUNC_ARCTAN2_F4F4F4, "arctan2_f4f4f4", atan2f, atan2f2, vsAtan2)
FUNC_F4F4F4(FUNC_F4F4F4_LAST,    NULL,          NULL,   NULL,    NULL)
#ifdef ELIDE_FUNC_F4F4F4
#undef ELIDE_FUNC_F4F4F4
#undef FUNC_F4F4F4
#endif

#ifndef FUNC_C8C8
#define ELIDE_FUNC_C8C8
#define FUNC_C8C8(...)
#endif
FUNC_C8C8(FUNC_SQRT_C8C8,    "sqrt_c8c8",     nx_sqrt, nx_sqrt, vcSqrt)
FUNC_C8C8(FUNC_SIN_C8C8,     "sin_c8c8",      nx_sin,   vcSin)
FUNC_C8C8(FUNC_COS_C8C8,     "cos_c8c8",      nx_cos,   vcCos)
FUNC_C8C8(FUNC_TAN_C8C8,     "tan_c8c8",      nx_tan,   vcTan)
FUNC_C8C8(FUNC_ARCSIN_C8C8,  "arcsin_c8c8",   nx_asin,  vcAsin)
FUNC_C8C8(FUNC_ARCCOS_C8C8,  "arccos_c8c8",   nx_acos,  vcAcos)
FUNC_C8C8(FUNC_ARCTAN_C8C8,  "arctan_c8c8",   nx_atan,  vcAtan)
FUNC_C8C8(FUNC_SINH_C8C8,    "sinh_c8c8",     nx_sinh,  vcSinh)
FUNC_C8C8(FUNC_COSH_C8C8,    "cosh_c8c8",     nx_cosh,  vcCosh)
FUNC_C8C8(FUNC_TANH_C8C8,    "tanh_c8c8",     nx_tanh,  vcTanh)
FUNC_C8C8(FUNC_ARCSINH_C8C8, "arcsinh_c8c8",  nx_asinh, vcAsinh)
FUNC_C8C8(FUNC_ARCCOSH_C8C8, "arccosh_c8c8",  nx_acosh, vcAcosh)
FUNC_C8C8(FUNC_ARCTANH_C8C8, "arctanh_c8c8",  nx_atanh, vcAtanh)
FUNC_C8C8(FUNC_LOG_C8C8,     "log_c8c8",      nx_log,   vcLn)
FUNC_C8C8(FUNC_LOG1P_C8C8,   "log1p_c8c8",    nx_log1p, vcLog1p)
FUNC_C8C8(FUNC_LOG10_C8C8,   "log10_c8c8",    nx_log10, vcLog10)
FUNC_C8C8(FUNC_EXP_C8C8,     "exp_c8c8",      nx_exp,   vcExp)
FUNC_C8C8(FUNC_EXPM1_C8C8,   "expm1_c8c8",    nx_expm1, vcExpm1)
FUNC_C8C8(FUNC_ABS_C8C8,     "absolute_c8c8", nx_abs,   vcAbs_)
FUNC_C8C8(FUNC_CONJ_C8C8,    "conjugate_c8c8",nx_conj,  vcConj)
FUNC_C8C8(FUNC_C8C8_LAST,    NULL,          NULL,     NULL)
#ifdef ELIDE_FUNC_C8C8
#undef ELIDE_FUNC_C8C8
#undef FUNC_C8C8
#endif

#ifndef FUNC_C8C8C8
#define ELIDE_FUNC_C8C8C8
#define FUNC_C8C8C8(...)
#endif
FUNC_C8C8C8(FUNC_POW_C8C8C8,   "pow_c8c8c8", nx_pow)
FUNC_C8C8C8(FUNC_C8C8C8_LAST,  NULL,      NULL)
#ifdef ELIDE_FUNC_C8C8C8
#undef ELIDE_FUNC_C8C8C8
#undef FUNC_C8C8C8
#endif

#ifndef FUNC_F8F8
#define ELIDE_FUNC_F8F8
#define FUNC_F8F8(...)
#endif
FUNC_F8F8(FUNC_SQRT_F8F8,    "sqrt_f8f8",     sqrt,  vdSqrt)
FUNC_F8F8(FUNC_SIN_F8F8,     "sin_f8f8",      sin,   vdSin)
FUNC_F8F8(FUNC_COS_F8F8,     "cos_f8f8",      cos,   vdCos)
FUNC_F8F8(FUNC_TAN_F8F8,     "tan_f8f8",      tan,   vdTan)
FUNC_F8F8(FUNC_ARCSIN_F8F8,  "arcsin_f8f8",   asin,  vdAsin)
FUNC_F8F8(FUNC_ARCCOS_F8F8,  "arccos_f8f8",   acos,  vdAcos)
FUNC_F8F8(FUNC_ARCTAN_F8F8,  "arctan_f8f8",   atan,  vdAtan)
FUNC_F8F8(FUNC_SINH_F8F8,    "sinh_f8f8",     sinh,  vdSinh)
FUNC_F8F8(FUNC_COSH_F8F8,    "cosh_f8f8",     cosh,  vdCosh)
FUNC_F8F8(FUNC_TANH_F8F8,    "tanh_f8f8",     tanh,  vdTanh)
FUNC_F8F8(FUNC_ARCSINH_F8F8, "arcsinh_f8f8",  asinh, vdAsinh)
FUNC_F8F8(FUNC_ARCCOSH_F8F8, "arccosh_f8f8",  acosh, vdAcosh)
FUNC_F8F8(FUNC_ARCTANH_F8F8, "arctanh_f8f8",  atanh, vdAtanh)
FUNC_F8F8(FUNC_LOG_F8F8,     "log_f8f8",      log,   vdLn)
FUNC_F8F8(FUNC_LOG1P_F8F8,   "log1p_f8f8",    log1p, vdLog1p)
FUNC_F8F8(FUNC_LOG10_F8F8,   "log10_f8f8",    log10, vdLog10)
FUNC_F8F8(FUNC_EXP_F8F8,     "exp_f8f8",      exp,   vdExp)
FUNC_F8F8(FUNC_EXPM1_F8F8,   "expm1_f8f8",    expm1, vdExpm1)
FUNC_F8F8(FUNC_ABS_F8F8,     "absolute_f8f8", fabs,  vdAbs)
FUNC_F8F8(FUNC_CONJ_F8F8,    "conjugate_f8f8",fconj, vdConj)
FUNC_F8F8(FUNC_F8F8_LAST,    NULL,          NULL,  NULL)
#ifdef ELIDE_FUNC_F8F8
#undef ELIDE_FUNC_F8F8
#undef FUNC_F8F8
#endif

#ifndef FUNC_F8F8F8
#define ELIDE_FUNC_F8F8F8
#define FUNC_F8F8F8(...)
#endif
FUNC_F8F8F8(FUNC_FMOD_F8F8F8,    "fmod_f8f8f8",    fmod,  vdfmod)
FUNC_F8F8F8(FUNC_ARCTAN2_F8F8F8, "arctan2_f8f8f8", atan2, vdAtan2)
FUNC_F8F8F8(FUNC_F8F8F8_LAST,    NULL,          NULL,  NULL)
#ifdef ELIDE_FUNC_F8F8F8
#undef ELIDE_FUNC_F8F8F8
#undef FUNC_F8F8F8
#endif

#ifndef FUNC_C16C16
#define ELIDE_FUNC_C16C16
#define FUNC_C16C16(...)
#endif
FUNC_C16C16(FUNC_SQRT_C16C16,    "sqrt_c16c16",     nc_sqrt,  vzSqrt)
FUNC_C16C16(FUNC_SIN_C16C16,     "sin_c16c16",      nc_sin,   vzSin)
FUNC_C16C16(FUNC_COS_C16C16,     "cos_c16c16",      nc_cos,   vzCos)
FUNC_C16C16(FUNC_TAN_C16C16,     "tan_c16c16",      nc_tan,   vzTan)
FUNC_C16C16(FUNC_ARCSIN_C16C16,  "arcsin_c16c16",   nc_asin,  vzAsin)
FUNC_C16C16(FUNC_ARCCOS_C16C16,  "arccos_c16c16",   nc_acos,  vzAcos)
FUNC_C16C16(FUNC_ARCTAN_C16C16,  "arctan_c16c16",   nc_atan,  vzAtan)
FUNC_C16C16(FUNC_SINH_C16C16,    "sinh_c16c16",     nc_sinh,  vzSinh)
FUNC_C16C16(FUNC_COSH_C16C16,    "cosh_c16c16",     nc_cosh,  vzCosh)
FUNC_C16C16(FUNC_TANH_C16C16,    "tanh_c16c16",     nc_tanh,  vzTanh)
FUNC_C16C16(FUNC_ARCSINH_C16C16, "arcsinh_c16c16",  nc_asinh, vzAsinh)
FUNC_C16C16(FUNC_ARCCOSH_C16C16, "arccosh_c16c16",  nc_acosh, vzAcosh)
FUNC_C16C16(FUNC_ARCTANH_C16C16, "arctanh_c16c16",  nc_atanh, vzAtanh)
FUNC_C16C16(FUNC_LOG_C16C16,     "log_c16c16",      nc_log,   vzLn)
FUNC_C16C16(FUNC_LOG1P_C16C16,   "log1p_c16c16",    nc_log1p, vzLog1p)
FUNC_C16C16(FUNC_LOG10_C16C16,   "log10_c16c16",    nc_log10, vzLog10)
FUNC_C16C16(FUNC_EXP_C16C16,     "exp_c16c16",      nc_exp,   vzExp)
FUNC_C16C16(FUNC_EXPM1_C16C16,   "expm1_c16c16",    nc_expm1, vzExpm1)
FUNC_C16C16(FUNC_ABS_C16C16,     "absolute_c16c16", nc_abs,   vzAbs_)
FUNC_C16C16(FUNC_CONJ_C16C16,    "conjugate_c16c16",nc_conj,  vzConj)
FUNC_C16C16(FUNC_C16C16_LAST,    NULL,          NULL,     NULL)
#ifdef ELIDE_FUNC_C16C16
#undef ELIDE_FUNC_C16C16
#undef FUNC_C16C16
#endif

#ifndef FUNC_C16C16C16
#define ELIDE_FUNC_C16C16C16
#define FUNC_C16C16C16(...)
#endif
FUNC_C16C16C16(FUNC_POW_C16C16C16,   "pow_c16c16c16", nc_pow)
FUNC_C16C16C16(FUNC_C16C16C16_LAST,  NULL,      NULL)
#ifdef ELIDE_FUNC_C16C16C16
#undef ELIDE_FUNC_C16C16C16
#undef FUNC_C16C16C16
#endif

