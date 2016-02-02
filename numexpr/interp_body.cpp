/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/


{
#define VEC_LOOP(expr) for(j = 0; j < BLOCK_SIZE; j++) {       \
        expr;                                   \
    }

#define VEC_ARG0(expr)                          \
    BOUNDS_CHECK(store_in);                     \
    {                                           \
        char *dest = mem[store_in];             \
        VEC_LOOP(expr);                         \
    } break

#define VEC_ARG1(expr)                          \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        npy_intp ss1 = params.memsizes[arg1];       \
        npy_intp sb1 = memsteps[arg1];              \
        /* nowarns is defined and used so as to \
        avoid compiler warnings about unused    \
        variables */                            \
        npy_intp nowarns = ss1+sb1+*x1;             \
        nowarns += 1;                           \
        VEC_LOOP(expr);                         \
    } break

#define VEC_ARG2(expr)                          \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    BOUNDS_CHECK(arg2);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        npy_intp ss1 = params.memsizes[arg1];       \
        npy_intp sb1 = memsteps[arg1];              \
        /* nowarns is defined and used so as to \
        avoid compiler warnings about unused    \
        variables */                            \
        npy_intp nowarns = ss1+sb1+*x1;             \
        char *x2 = mem[arg2];                   \
        npy_intp ss2 = params.memsizes[arg2];       \
        npy_intp sb2 = memsteps[arg2];              \
        nowarns += ss2+sb2+*x2;                 \
        VEC_LOOP(expr);                         \
    } break

#define VEC_ARG3(expr)                          \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    BOUNDS_CHECK(arg2);                         \
    BOUNDS_CHECK(arg3);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        npy_intp ss1 = params.memsizes[arg1];       \
        npy_intp sb1 = memsteps[arg1];              \
        /* nowarns is defined and used so as to \
        avoid compiler warnings about unused    \
        variables */                            \
        npy_intp nowarns = ss1+sb1+*x1;             \
        char *x2 = mem[arg2];                   \
        npy_intp ss2 = params.memsizes[arg2];       \
        npy_intp sb2 = memsteps[arg2];              \
        char *x3 = mem[arg3];                   \
        npy_intp ss3 = params.memsizes[arg3];       \
        npy_intp sb3 = memsteps[arg3];              \
        nowarns += ss2+sb2+*x2;                 \
        nowarns += ss3+sb3+*x3;                 \
        VEC_LOOP(expr);                         \
    } break

#define VEC_ARG1_VML(expr)                      \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        expr;                                   \
    } break

#define VEC_ARG2_VML(expr)                      \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    BOUNDS_CHECK(arg2);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        char *x2 = mem[arg2];                   \
        expr;                                   \
    } break

#define VEC_ARG3_VML(expr)                      \
    BOUNDS_CHECK(store_in);                     \
    BOUNDS_CHECK(arg1);                         \
    BOUNDS_CHECK(arg2);                         \
    BOUNDS_CHECK(arg3);                         \
    {                                           \
        char *dest = mem[store_in];             \
        char *x1 = mem[arg1];                   \
        char *x2 = mem[arg2];                   \
        char *x3 = mem[arg3];                   \
        expr;                                   \
    } break

    int pc;
    unsigned int j;

    // set up pointers to next block of inputs and outputs
#ifdef SINGLE_ITEM_CONST_LOOP
    mem[0] = params.output;
#else // SINGLE_ITEM_CONST_LOOP
    // use the iterator's inner loop data
    memcpy(mem, iter_dataptr, (1+params.n_inputs)*sizeof(char*));
#  ifndef NO_OUTPUT_BUFFERING
    // if output buffering is necessary, first write to the buffer
    if(params.out_buffer != NULL) {
        mem[0] = params.out_buffer;
    }
#  endif // NO_OUTPUT_BUFFERING
    memcpy(memsteps, iter_strides, (1+params.n_inputs)*sizeof(npy_intp));
#endif // SINGLE_ITEM_CONST_LOOP

    // WARNING: From now on, only do references to mem[arg[123]]
    // & memsteps[arg[123]] inside the VEC_ARG[123] macros,
    // or you will risk accessing invalid addresses.

    for (pc = 0; pc < params.prog_len; pc += 4) {
        unsigned char op = params.program[pc];
        unsigned int store_in = params.program[pc+1];
        unsigned int arg1 = params.program[pc+2];
        unsigned int arg2 = params.program[pc+3];
        #define      arg3   params.program[pc+5]
        // Iterator reduce macros
#ifdef REDUCTION_INNER_LOOP // Reduce is the inner loop
        #define i4_reduce    *(npy_int32 *)dest
        #define i8_reduce    *(npy_int64 *)dest
        #define f4_reduce    *(npy_float32 *)dest
        #define f8_reduce    *(npy_float64 *)dest
        #define c16r_reduce   *(npy_float64 *)dest
        #define c16i_reduce   *((npy_float64 *)dest+1)
        #define c8r_reduce   *(npy_float32 *)dest
        #define c8i_reduce   *((npy_float32 *)dest+1)
#else /* Reduce is the outer loop */
        #define i4_reduce    i4_dest
        #define i8_reduce    i8_dest
        #define f4_reduce    f4_dest
        #define f8_reduce    f8_dest
        #define c16r_reduce   c16r_dest
        #define c16i_reduce   c16i_dest
        #define c8r_reduce   c8r_dest
        #define c8i_reduce   c8i_dest
#endif
        #define b1_dest ((npy_bool *)dest)[j]
        #define i4_dest ((npy_int32 *)dest)[j]
        #define i8_dest ((npy_int64 *)dest)[j]
        #define f4_dest ((npy_float32 *)dest)[j]
        #define f8_dest ((npy_float64 *)dest)[j]
        #define c16r_dest ((npy_float64 *)dest)[2*j]
        #define c16i_dest ((npy_float64 *)dest)[2*j+1]
        #define c8r_dest ((npy_float32 *)dest)[2*j]
        #define c8i_dest ((npy_float32 *)dest)[2*j+1]
        #define s1_dest ((char *)dest + j*memsteps[store_in])
        #define b1_1    ((char   *)(x1+j*sb1))[0]
        #define i4_1    ((npy_int32    *)(x1+j*sb1))[0]
        #define i8_1    ((npy_int64 *)(x1+j*sb1))[0]
        #define f4_1    ((npy_float32  *)(x1+j*sb1))[0]
        #define f8_1    ((npy_float64 *)(x1+j*sb1))[0]
        #define c16_1r   ((npy_float64 *)(x1+j*sb1))[0]
        #define c16_1i   ((npy_float64 *)(x1+j*sb1))[1]
        #define c8_1r   ((npy_float32 *)(x1+j*sb1))[0]
        #define c8_1i   ((npy_float32 *)(x1+j*sb1))[1]
        #define s1_1    ((char   *)x1+j*sb1)
        #define b1_2    ((npy_bool   *)(x2+j*sb2))[0]
        #define i4_2    ((npy_int32    *)(x2+j*sb2))[0]
        #define i8_2    ((npy_int64 *)(x2+j*sb2))[0]
        #define f4_2    ((npy_float32  *)(x2+j*sb2))[0]
        #define f8_2    ((npy_float64 *)(x2+j*sb2))[0]
        #define c16_2r   ((npy_float64 *)(x2+j*sb2))[0]
        #define c16_2i   ((npy_float64 *)(x2+j*sb2))[1]
        #define c8_2r   ((npy_float32 *)(x2+j*sb2))[0]
        #define c8_2i   ((npy_float32 *)(x2+j*sb2))[1]
        #define s1_2    ((char   *)x2+j*sb2)
        #define b1_3    ((npy_bool   *)(x3+j*sb3))[0]
        #define i4_3    ((npy_int32    *)(x3+j*sb3))[0]
        #define i8_3    ((npy_int64 *)(x3+j*sb3))[0]
        #define f4_3    ((npy_float32  *)(x3+j*sb3))[0]
        #define f8_3    ((npy_float64 *)(x3+j*sb3))[0]
        #define c16_3r   ((npy_float64 *)(x3+j*sb3))[0]
        #define c16_3i   ((npy_float64 *)(x3+j*sb3))[1]
        #define c8_3r   ((npy_float32 *)(x3+j*sb3))[0]
        #define c8_3i   ((npy_float32 *)(x3+j*sb3))[1]
        #define s1_3    ((char   *)x3+j*sb3)
        /* Some temporaries */
        double da, db;
        npy_cdouble ca, cb;
        float fa, fb;
        npy_cfloat xa, xb;

        switch (op) {

        case OP_NOOP: break;

        case OP_COPY_B1B1: VEC_ARG1(b1_dest = b1_1);
        case OP_COPY_S1S1: VEC_ARG1(memcpy(s1_dest, s1_1, ss1));
        /* The next versions of copy opcodes can cope with unaligned
           data even on platforms that crash while accessing it
           (like the Sparc architecture under Solaris). */
        case OP_COPY_I4I4: VEC_ARG1(memcpy(&i4_dest, s1_1, sizeof(npy_int32)));
        case OP_COPY_I8I8: VEC_ARG1(memcpy(&i8_dest, s1_1, sizeof(npy_int64)));
        case OP_COPY_F4F4: VEC_ARG1(memcpy(&f4_dest, s1_1, sizeof(npy_float32)));
        case OP_COPY_F8F8: VEC_ARG1(memcpy(&f8_dest, s1_1, sizeof(npy_float64)));
        case OP_COPY_C16C16: VEC_ARG1(memcpy(&c16r_dest, s1_1, sizeof(npy_complex64)));
        case OP_COPY_C8C8: VEC_ARG1(memcpy(&c8r_dest, s1_1, sizeof(npy_complex128)));

        /* Bool */
        case OP_INVERT_B1B1: VEC_ARG1(b1_dest = !b1_1);
        case OP_AND_B1B1B1: VEC_ARG2(b1_dest = (b1_1 && b1_2));
        case OP_OR_B1B1B1: VEC_ARG2(b1_dest = (b1_1 || b1_2));

        case OP_EQ_B1B1B1: VEC_ARG2(b1_dest = (b1_1 == b1_2));
        case OP_NE_B1B1B1: VEC_ARG2(b1_dest = (b1_1 != b1_2));
        case OP_WHERE_B1B1B1B1: VEC_ARG3(b1_dest = b1_1 ? b1_2 : b1_3);

        /* Comparisons */
        case OP_GT_B1I4I4: VEC_ARG2(b1_dest = (i4_1 > i4_2));
        case OP_GE_B1I4I4: VEC_ARG2(b1_dest = (i4_1 >= i4_2));
        case OP_EQ_B1I4I4: VEC_ARG2(b1_dest = (i4_1 == i4_2));
        case OP_NE_B1I4I4: VEC_ARG2(b1_dest = (i4_1 != i4_2));

        case OP_GT_B1I8I8: VEC_ARG2(b1_dest = (i8_1 > i8_2));
        case OP_GE_B1I8I8: VEC_ARG2(b1_dest = (i8_1 >= i8_2));
        case OP_EQ_B1I8I8: VEC_ARG2(b1_dest = (i8_1 == i8_2));
        case OP_NE_B1I8I8: VEC_ARG2(b1_dest = (i8_1 != i8_2));

        case OP_GT_B1F4F4: VEC_ARG2(b1_dest = (f4_1 > f4_2));
        case OP_GE_B1F4F4: VEC_ARG2(b1_dest = (f4_1 >= f4_2));
        case OP_EQ_B1F4F4: VEC_ARG2(b1_dest = (f4_1 == f4_2));
        case OP_NE_B1F4F4: VEC_ARG2(b1_dest = (f4_1 != f4_2));

        case OP_GT_B1F8F8: VEC_ARG2(b1_dest = (f8_1 > f8_2));
        case OP_GE_B1F8F8: VEC_ARG2(b1_dest = (f8_1 >= f8_2));
        case OP_EQ_B1F8F8: VEC_ARG2(b1_dest = (f8_1 == f8_2));
        case OP_NE_B1F8F8: VEC_ARG2(b1_dest = (f8_1 != f8_2));

        case OP_GT_B1S1S1: VEC_ARG2(b1_dest = (stringcmp(s1_1, s1_2, ss1, ss2) > 0));
        case OP_GE_B1S1S1: VEC_ARG2(b1_dest = (stringcmp(s1_1, s1_2, ss1, ss2) >= 0));
        case OP_EQ_B1S1S1: VEC_ARG2(b1_dest = (stringcmp(s1_1, s1_2, ss1, ss2) == 0));
        case OP_NE_B1S1S1: VEC_ARG2(b1_dest = (stringcmp(s1_1, s1_2, ss1, ss2) != 0));

        case OP_CONTAINS_B1S1S1: VEC_ARG2(b1_dest = stringcontains(s1_1, s1_2, ss1, ss2));

        /* Int */
        case OP_CAST_I4B1: VEC_ARG1(i4_dest = (npy_int32)(b1_1));
        case OP_ONES_LIKE_I4I4: VEC_ARG0(i4_dest = 1);
        case OP_NEG_I4I4: VEC_ARG1(i4_dest = -i4_1);

        case OP_ADD_I4I4I4: VEC_ARG2(i4_dest = i4_1 + i4_2);
        case OP_SUB_I4I4I4: VEC_ARG2(i4_dest = i4_1 - i4_2);
        case OP_MUL_I4I4I4: VEC_ARG2(i4_dest = i4_1 * i4_2);
        case OP_DIV_I4I4I4: VEC_ARG2(i4_dest = i4_2 ? (i4_1 / i4_2) : 0);
        case OP_POW_I4I4I4: VEC_ARG2(i4_dest = (i4_2 < 0) ? (1 / i4_1) : (npy_int32)pow((npy_float32)i4_1, i4_2));
        case OP_MOD_I4I4I4: VEC_ARG2(i4_dest = i4_2 ? (i4_1 % i4_2) : 0);
        case OP_LSHIFT_I4I4I4: VEC_ARG2(i4_dest = i4_1 << i4_2);
        case OP_RSHIFT_I4I4I4: VEC_ARG2(i4_dest = i4_1 >> i4_2);

        case OP_WHERE_I4B1I4I4: VEC_ARG3(i4_dest = b1_1 ? i4_2 : i4_3);

        /* Long */
        case OP_CAST_I8I4: VEC_ARG1(i8_dest = (npy_int64)(i4_1));
        case OP_ONES_LIKE_I8I8: VEC_ARG0(i8_dest = 1);
        case OP_NEG_I8I8: VEC_ARG1(i8_dest = -i8_1);

        case OP_ADD_I8I8I8: VEC_ARG2(i8_dest = i8_1 + i8_2);
        case OP_SUB_I8I8I8: VEC_ARG2(i8_dest = i8_1 - i8_2);
        case OP_MUL_I8I8I8: VEC_ARG2(i8_dest = i8_1 * i8_2);
        case OP_DIV_I8I8I8: VEC_ARG2(i8_dest = i8_2 ? (i8_1 / i8_2) : 0);
        case OP_POW_I8I8I8: VEC_ARG2(i8_dest = (i8_2 < 0) ? (1 / i8_1) : (npy_int64)pow((npy_float64)i8_1, (npy_float64)i8_2));
        case OP_MOD_I8I8I8: VEC_ARG2(i8_dest = i8_2 ? (i8_1 % i8_2) : 0);
        case OP_LSHIFT_I8I8I8: VEC_ARG2(i8_dest = i8_1 << i8_2);
        case OP_RSHIFT_I8I8I8: VEC_ARG2(i8_dest = i8_1 >> i8_2);

        case OP_WHERE_I8B1I8I8: VEC_ARG3(i8_dest = b1_1 ? i8_2 : i8_3);

        /* Float */
        case OP_CAST_F4I4: VEC_ARG1(f4_dest = (npy_float32)(i4_1));
        case OP_CAST_F4I8: VEC_ARG1(f4_dest = (npy_float32)(i8_1));
        case OP_ONES_LIKE_F4F4: VEC_ARG0(f4_dest = 1.0);
        case OP_NEG_F4F4: VEC_ARG1(f4_dest = -f4_1);

        case OP_ADD_F4F4F4: VEC_ARG2(f4_dest = f4_1 + f4_2);
        case OP_SUB_F4F4F4: VEC_ARG2(f4_dest = f4_1 - f4_2);
        case OP_MUL_F4F4F4: VEC_ARG2(f4_dest = f4_1 * f4_2);
        case OP_DIV_F4F4F4:
#ifdef USE_VML
            VEC_ARG2_VML(vsDiv(BLOCK_SIZE,
                               (npy_float32*)x1, (npy_float32*)x2, (npy_float32*)dest));
#else
            VEC_ARG2(f4_dest = f4_1 / f4_2);
#endif
        case OP_POW_F4F4F4:
#ifdef USE_VML
            VEC_ARG2_VML(vsPow(BLOCK_SIZE,
                               (npy_float32*)x1, (npy_float32*)x2, (npy_float32*)dest));
#else
            VEC_ARG2(f4_dest = powf(f4_1, f4_2));
#endif
        case OP_MOD_F4F4F4: VEC_ARG2(f4_dest = f4_1 - floorf(f4_1/f4_2) * f4_2);

        case OP_SQRT_F4F4:
#ifdef USE_VML
            VEC_ARG1_VML(vsSqrt(BLOCK_SIZE, (npy_float32*)x1, (npy_float32*)dest));
#else
            VEC_ARG1(f4_dest = sqrtf(f4_1));
#endif

        case OP_WHERE_F4B1F4F4: VEC_ARG3(f4_dest = b1_1 ? f4_2 : f4_3);

        case OP_FUNC_F4F4N0:
#ifdef USE_VML
            VEC_ARG1_VML(functions_ff_vml[arg2](BLOCK_SIZE,
                                                (npy_float32*)x1, (npy_float32*)dest));
#else
            VEC_ARG1(f4_dest = functions_f4f4[arg2](f4_1));
#endif
        case OP_FUNC_F4F4F4N0:
#ifdef USE_VML
            VEC_ARG2_VML(functions_fff_vml[arg3](BLOCK_SIZE,
                                                 (npy_float32*)x1, (npy_float32*)x2,
                                                 (npy_float32*)dest));
#else
            VEC_ARG2(f4_dest = functions_f4f4f4[arg3](f4_1, f4_2));
#endif

        /* Double */
        case OP_CAST_F8I4: VEC_ARG1(f8_dest = (npy_float64)(i4_1));
        case OP_CAST_F8I8: VEC_ARG1(f8_dest = (npy_float64)(i8_1));
        case OP_CAST_F8F4: VEC_ARG1(f8_dest = (npy_float64)(f4_1));
        case OP_ONES_LIKE_F8F8: VEC_ARG0(f8_dest = 1.0);
        case OP_NEG_F8F8: VEC_ARG1(f8_dest = -f8_1);

        case OP_ADD_F8F8F8: VEC_ARG2(f8_dest = f8_1 + f8_2);
        case OP_SUB_F8F8F8: VEC_ARG2(f8_dest = f8_1 - f8_2);
        case OP_MUL_F8F8F8: VEC_ARG2(f8_dest = f8_1 * f8_2);
        case OP_DIV_F8F8F8:
#ifdef USE_VML
            VEC_ARG2_VML(vdDiv(BLOCK_SIZE,
                               (npy_float64*)x1, (npy_float64*)x2, (npy_float64*)dest));
#else
            VEC_ARG2(f8_dest = f8_1 / f8_2);
#endif
        case OP_POW_F8F8F8:
#ifdef USE_VML
            VEC_ARG2_VML(vdPow(BLOCK_SIZE,
                               (npy_float64*)x1, (npy_float64*)x2, (npy_float64*)dest));
#else
            VEC_ARG2(f8_dest = pow(f8_1, f8_2));
#endif
        case OP_MOD_F8F8F8: VEC_ARG2(f8_dest = f8_1 - floor(f8_1/f8_2) * f8_2);

        case OP_SQRT_F8F8:
#ifdef USE_VML
            VEC_ARG1_VML(vdSqrt(BLOCK_SIZE, (npy_float64*)x1, (npy_float64*)dest));
#else
            VEC_ARG1(f8_dest = sqrt(f8_1));
#endif

        case OP_WHERE_F8B1F8F8: VEC_ARG3(f8_dest = b1_1 ? f8_2 : f8_3);

        case OP_FUNC_F8F8N0:
#ifdef USE_VML
            VEC_ARG1_VML(functions_dd_vml[arg2](BLOCK_SIZE,
                                                (npy_float64*)x1, (npy_float64*)dest));
#else
            VEC_ARG1(f8_dest = functions_f8f8[arg2](f8_1));
#endif
        case OP_FUNC_F8F8F8N0:
#ifdef USE_VML
            VEC_ARG2_VML(functions_ddd_vml[arg3](BLOCK_SIZE,
                                                 (npy_float64*)x1, (npy_float64*)x2,
                                                 (npy_float64*)dest));
#else
            VEC_ARG2(f8_dest = functions_f8f8f8[arg3](f8_1, f8_2));
#endif

        /* Complex double */
        case OP_CAST_C16I4: VEC_ARG1(c16r_dest = (npy_float64)(i4_1);
                                  c16i_dest = 0);
        case OP_CAST_C16I8: VEC_ARG1(c16r_dest = (npy_float64)(i8_1);
                                  c16i_dest = 0);
        case OP_CAST_C16F4: VEC_ARG1(c16r_dest = f4_1;
                                  c16i_dest = 0);
        case OP_CAST_C16F8: VEC_ARG1(c16r_dest = f8_1;
                                  c16i_dest = 0);
        case OP_ONES_LIKE_C16C16: VEC_ARG0(c16r_dest = 1;
                                       c16i_dest = 0);
        case OP_NEG_C16C16: VEC_ARG1(c16r_dest = -c16_1r;
                                 c16i_dest = -c16_1i);

        case OP_ADD_C16C16C16: VEC_ARG2(c16r_dest = c16_1r + c16_2r;
                                  c16i_dest = c16_1i + c16_2i);
        case OP_SUB_C16C16C16: VEC_ARG2(c16r_dest = c16_1r - c16_2r;
                                  c16i_dest = c16_1i - c16_2i);
        case OP_MUL_C16C16C16: VEC_ARG2(da = c16_1r*c16_2r - c16_1i*c16_2i;
                                  c16i_dest = c16_1r*c16_2i + c16_1i*c16_2r;
                                  c16r_dest = da);
        case OP_DIV_C16C16C16:
#ifdef USE_VMLXXX /* VML complex division is slower */
            VEC_ARG2_VML(vzDiv(BLOCK_SIZE, (const MKL_Complex16*)x1,
                               (const MKL_Complex16*)x2, (MKL_Complex16*)dest));
#else
            VEC_ARG2(da = c16_2r*c16_2r + c16_2i*c16_2i;
                     db = (c16_1r*c16_2r + c16_1i*c16_2i) / da;
                     c16i_dest = (c16_1i*c16_2r - c16_1r*c16_2i) / da;
                     c16r_dest = db);
#endif
        case OP_EQ_B1C16C16: VEC_ARG2(b1_dest = (c16_1r == c16_2r && c16_1i == c16_2i));
        case OP_NE_B1C16C16: VEC_ARG2(b1_dest = (c16_1r != c16_2r || c16_1i != c16_2i));

        case OP_WHERE_C16B1C16C16: VEC_ARG3(c16r_dest = b1_1 ? c16_2r : c16_3r;
                                     c16i_dest = b1_1 ? c16_2i : c16_3i);
        case OP_FUNC_C16C16N0:
#ifdef USE_VML
            VEC_ARG1_VML(functions_cc_vml[arg2](BLOCK_SIZE,
                                                (const MKL_Complex16*)x1,
                                                (MKL_Complex16*)dest));
#else
            VEC_ARG1(ca.real = c16_1r;
                     ca.imag = c16_1i;
                     functions_c16c16[arg2](&ca, &ca);
                     c16r_dest = ca.real;
                     c16i_dest = ca.imag);
#endif
        case OP_FUNC_C16C16C16N0: VEC_ARG2(ca.real = c16_1r;
                                    ca.imag = c16_1i;
                                    cb.real = c16_2r;
                                    cb.imag = c16_2i;
                                    functions_c16c16c16[arg3](&ca, &cb, &ca);
                                    c16r_dest = ca.real;
                                    c16i_dest = ca.imag);

        case OP_REAL_F8C16: VEC_ARG1(f8_dest = c16_1r);
        case OP_IMAG_F8C16: VEC_ARG1(f8_dest = c16_1i);
        case OP_COMPLEX_C16F8F8: VEC_ARG2(c16r_dest = f8_1;
                                      c16i_dest = f8_2);

        /* Complex float */
        case OP_CAST_C8I4: VEC_ARG1(c8r_dest = (npy_float32)(i4_1);
                                  c8i_dest = 0);
        case OP_CAST_C8I8: VEC_ARG1(c8r_dest = (npy_float32)(i8_1);
                                  c8i_dest = 0);
        case OP_CAST_C8F4: VEC_ARG1(c8r_dest = f4_1;
                                  c8i_dest = 0);
        // RAM: this needs a downcast
        case OP_CAST_C8F8: VEC_ARG1(c8r_dest = (npy_float32)f8_1;
                                  c8i_dest = 0);
        case OP_ONES_LIKE_C8C8: VEC_ARG0(c8r_dest = 1;
                                       c8i_dest = 0);
        case OP_NEG_C8C8: VEC_ARG1(c8r_dest = -c8_1r;
                                 c8i_dest = -c8_1i);

        case OP_ADD_C8C8C8: VEC_ARG2(c8r_dest = c8_1r + c8_2r;
                                  c8i_dest = c8_1i + c8_2i);
        case OP_SUB_C8C8C8: VEC_ARG2(c8r_dest = c8_1r - c8_2r;
                                  c8i_dest = c8_1i - c8_2i);
        case OP_MUL_C8C8C8: VEC_ARG2(fa = c8_1r*c8_2r - c8_1i*c8_2i;
                                  c8i_dest = c8_1r*c8_2i + c8_1i*c8_2r;
                                  c8r_dest = fa);
        case OP_DIV_C8C8C8:
#ifdef USE_VMLXXX /* VML complex division is slower */
            VEC_ARG2_VML(vcDiv(BLOCK_SIZE, (const MKL_Complex8*)x1,
                               (const MKL_Complex8*)x2, (MKL_Complex8*)dest));
#else
            VEC_ARG2(fa = c8_2r*c8_2r + c8_2i*c8_2i;
                     fb = (c8_1r*c8_2r + c8_1i*c8_2i) / fa;
                     c8i_dest = (c8_1i*c8_2r - c8_1r*c8_2i) / fa;
                     c8r_dest = fb);
#endif
        case OP_EQ_B1C8C8: VEC_ARG2(b1_dest = (c8_1r == c8_2r && c8_1i == c8_2i));
        case OP_NE_B1C8C8: VEC_ARG2(b1_dest = (c8_1r != c8_2r || c8_1i != c8_2i));

        case OP_WHERE_C8B1C8C8: VEC_ARG3(c8r_dest = b1_1 ? c8_2r : c8_3r;
                                     c8i_dest = b1_1 ? c8_2i : c8_3i);
        
        case OP_FUNC_C8C8N0:
#ifdef USE_VML
            VEC_ARG1_VML(functions_xx_vml[arg2](BLOCK_SIZE,
                                                (const MKL_Complex8*)x1,
                                                (MKL_Complex8*)dest));
#else
            // RAM: Ok something in here is not right...
            VEC_ARG1(xa.real = c8_1r;
                     xa.imag = c8_1i;
                     functions_c8c8[arg2](&xa, &xa);
                     c8r_dest = xa.real;
                     c8i_dest = xa.imag);
#endif
        case OP_FUNC_C8C8C8N0: VEC_ARG2(xa.real = c8_1r;
                                    xa.imag = c8_1i;
                                    xb.real = c8_2r;
                                    xb.imag = c8_2i;
                                    functions_c8c8c8[arg3](&xa, &xb, &xa);
                                    c8r_dest = xa.real;
                                    c8i_dest = xa.imag);

        case OP_REAL_F4C8: VEC_ARG1(f4_dest = c8_1r);
        case OP_IMAG_F4C8: VEC_ARG1(f4_dest = c8_1i);
        case OP_COMPLEX_C8F4F4: VEC_ARG2(c8r_dest = f8_1;
                                      c8i_dest = f8_2);

        /* Reductions */
        case OP_SUM_I4I4N0: VEC_ARG1(i4_reduce += i4_1);
        case OP_SUM_I8I8N0: VEC_ARG1(i8_reduce += i8_1);
        case OP_SUM_F4F4N0: VEC_ARG1(f4_reduce += f4_1);
        case OP_SUM_F8F8N0: VEC_ARG1(f8_reduce += f8_1);
        case OP_SUM_C16C16N0: VEC_ARG1(c16r_reduce += c16_1r;
                                  c16i_reduce += c16_1i);
        case OP_SUM_C8C8N0: VEC_ARG1(c8r_reduce += c8_1r;
                                  c8i_reduce += c8_1i);

        case OP_PROD_I4I4N0: VEC_ARG1(i4_reduce *= i4_1);
        case OP_PROD_I8I8N0: VEC_ARG1(i8_reduce *= i8_1);
        case OP_PROD_F4F4N0: VEC_ARG1(f4_reduce *= f4_1);
        case OP_PROD_F8F8N0: VEC_ARG1(f8_reduce *= f8_1);
        case OP_PROD_C16C16N0: VEC_ARG1(da = c16r_reduce*c16_1r - c16i_reduce*c16_1i;
                                   c16i_reduce = c16r_reduce*c16_1i + c16i_reduce*c16_1r;
                                   c16r_reduce = da);
        case OP_PROD_C8C8N0: VEC_ARG1(fa = c8r_reduce*c8_1r - c8i_reduce*c8_1i;
                                   c8i_reduce = c8r_reduce*c8_1i + c8i_reduce*c8_1r;
                                   c8r_reduce = fa);

        default:
            *pc_error = pc;
            return -3;
            break;
        }
    }


#ifndef NO_OUTPUT_BUFFERING
    // If output buffering was necessary, copy the buffer to the output
    if(params.out_buffer != NULL) {
        memcpy(iter_dataptr[0], params.out_buffer, params.memsizes[0] * BLOCK_SIZE);
    }
#endif // NO_OUTPUT_BUFFERING

#undef VEC_LOOP
#undef VEC_ARG1
#undef VEC_ARG2
#undef VEC_ARG3

#undef i4_reduce
#undef i8_reduce
#undef f4_reduce
#undef f8_reduce
#undef c16r_reduce
#undef c16i_reduce
#undef c8r_reduce
#undef c8i_reduce
#undef b1_dest
#undef i4_dest
#undef i8_dest
#undef f4_dest
#undef f8_dest
#undef c16r_dest
#undef c16i_dest
#undef c8r_dest
#undef c8i_dest
#undef s1_dest
#undef b1_1
#undef i4_1
#undef i8_1
#undef f4_1
#undef f8_1
#undef c16_1r
#undef c16_1i
#undef c8_1r
#undef c8_1i
#undef s1_1
#undef b1_2
#undef i4_2
#undef i8_2
#undef f4_2
#undef f8_2
#undef c16_2r
#undef c16_2i
#undef c8_2r
#undef c8_2i
#undef s1_2
#undef b1_3
#undef i4_3
#undef i8_3
#undef f4_3
#undef f8_3
#undef c16_3r
#undef c16_3i
#undef c8_3r
#undef c8_3i
#undef s1_3
}

/*
Local Variables:
   c-basic-offset: 4
End:
*/
