// WARNING: THIS FILE IS AUTO-GENERATED PRIOR TO COMPILATION.
// Editing should be done on the associated stub file instead.
/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: BSD
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

{
    npy_intp pc;

    // set up pointers to next block of inputs and outputs
#ifdef SINGLE_ITEM_CONST_LOOP
    //mem[0] = params->output;
#else // not SINGLE_ITEM_CONST_LOOP
    npy_intp J;
    // use the iterator's inner loop data
    // TODO: this is getting expensive to set mems for each block...
    npy_intp arrayCnt = 0;
    for( J = 0; J < params->n_reg; J++ ) {
        if( params->registers[J].kind == KIND_ARRAY ) {
            params->registers[J].mem = iterDataPtr[arrayCnt];
            params->registers[J].stride = iterStrides[arrayCnt];
            arrayCnt++;
        }
    }
#  ifndef NO_OUTPUT_BUFFERING
    // if output buffering is necessary, first write to the buffer
    // DISABLE OUTPUT BUFFERING
//    if(params->outBuffer != NULL) {
//        GET_RETURN_REG(params).mem = params->outBuffer;
//        //mem[0] = params->outBuffer;
//    }
#  endif // NO_OUTPUT_BUFFERING

#endif // not SINGLE_ITEM_CONST_LOOP

    // WARNING: From now on, only do references to mem[arg[123]]
    // & memsteps[arg[123]] inside the VEC_ARG[123] macros,
    // or you will risk accessing invalid addresses.
    
    // For the strings, I think they are actual string objects?
    // https://github.com/numpy/numpy/blob/c90d7c94fd2077d0beca48fa89a423da2b0bb663/numpy/core/include/numpy/npy_3kcompat.h

    for (pc = 0; pc < params->program_len; pc++ ) {

        // Sample debug output: 
        // printf( "Arg1(%d@%p) + ARG2(%d@%p) => DEST(%d@%p)\n", arg1, x1, arg2, x2, store_in, dest );
        
        // TODO: BLOCK_SIZE1 is the number of operations, not the array block size,
        // so the memory block scales with itemsize...
        //printf( "Exec op: %d\n", op );
        switch (params->program[pc].op) {
        case 0: 
            break;
        case 1: cast_11( block_size, pc, params ); break;
case 2: cast_b1( block_size, pc, params ); break;
case 3: cast_h1( block_size, pc, params ); break;
case 4: cast_l1( block_size, pc, params ); break;
case 5: cast_q1( block_size, pc, params ); break;
case 6: cast_B1( block_size, pc, params ); break;
case 7: cast_H1( block_size, pc, params ); break;
case 8: cast_L1( block_size, pc, params ); break;
case 9: cast_Q1( block_size, pc, params ); break;
case 10: cast_f1( block_size, pc, params ); break;
case 11: cast_d1( block_size, pc, params ); break;
case 12: cast_bb( block_size, pc, params ); break;
case 13: cast_hb( block_size, pc, params ); break;
case 14: cast_lb( block_size, pc, params ); break;
case 15: cast_qb( block_size, pc, params ); break;
case 16: cast_fb( block_size, pc, params ); break;
case 17: cast_db( block_size, pc, params ); break;
case 18: cast_hh( block_size, pc, params ); break;
case 19: cast_lh( block_size, pc, params ); break;
case 20: cast_qh( block_size, pc, params ); break;
case 21: cast_fh( block_size, pc, params ); break;
case 22: cast_dh( block_size, pc, params ); break;
case 23: cast_ll( block_size, pc, params ); break;
case 24: cast_ql( block_size, pc, params ); break;
case 25: cast_dl( block_size, pc, params ); break;
case 26: cast_qq( block_size, pc, params ); break;
case 27: cast_dq( block_size, pc, params ); break;
case 28: cast_hB( block_size, pc, params ); break;
case 29: cast_lB( block_size, pc, params ); break;
case 30: cast_qB( block_size, pc, params ); break;
case 31: cast_BB( block_size, pc, params ); break;
case 32: cast_HB( block_size, pc, params ); break;
case 33: cast_LB( block_size, pc, params ); break;
case 34: cast_QB( block_size, pc, params ); break;
case 35: cast_fB( block_size, pc, params ); break;
case 36: cast_dB( block_size, pc, params ); break;
case 37: cast_lH( block_size, pc, params ); break;
case 38: cast_qH( block_size, pc, params ); break;
case 39: cast_HH( block_size, pc, params ); break;
case 40: cast_LH( block_size, pc, params ); break;
case 41: cast_QH( block_size, pc, params ); break;
case 42: cast_fH( block_size, pc, params ); break;
case 43: cast_dH( block_size, pc, params ); break;
case 44: cast_qL( block_size, pc, params ); break;
case 45: cast_LL( block_size, pc, params ); break;
case 46: cast_QL( block_size, pc, params ); break;
case 47: cast_dL( block_size, pc, params ); break;
case 48: cast_QQ( block_size, pc, params ); break;
case 49: cast_dQ( block_size, pc, params ); break;
case 50: cast_ff( block_size, pc, params ); break;
case 51: cast_df( block_size, pc, params ); break;
case 52: cast_dd( block_size, pc, params ); break;
case 53: copy_11( block_size, pc, params ); break;
case 54: copy_bb( block_size, pc, params ); break;
case 55: copy_hh( block_size, pc, params ); break;
case 56: copy_ll( block_size, pc, params ); break;
case 57: copy_qq( block_size, pc, params ); break;
case 58: copy_BB( block_size, pc, params ); break;
case 59: copy_HH( block_size, pc, params ); break;
case 60: copy_LL( block_size, pc, params ); break;
case 61: copy_QQ( block_size, pc, params ); break;
case 62: copy_ff( block_size, pc, params ); break;
case 63: copy_dd( block_size, pc, params ); break;
case 64: copy_FF( block_size, pc, params ); break;
case 65: copy_DD( block_size, pc, params ); break;
case 66: add_111( block_size, pc, params ); break;
case 67: add_bbb( block_size, pc, params ); break;
case 68: add_hhh( block_size, pc, params ); break;
case 69: add_lll( block_size, pc, params ); break;
case 70: add_qqq( block_size, pc, params ); break;
case 71: add_BBB( block_size, pc, params ); break;
case 72: add_HHH( block_size, pc, params ); break;
case 73: add_LLL( block_size, pc, params ); break;
case 74: add_QQQ( block_size, pc, params ); break;
case 75: add_fff( block_size, pc, params ); break;
case 76: add_ddd( block_size, pc, params ); break;
case 77: sub_111( block_size, pc, params ); break;
case 78: sub_bbb( block_size, pc, params ); break;
case 79: sub_hhh( block_size, pc, params ); break;
case 80: sub_lll( block_size, pc, params ); break;
case 81: sub_qqq( block_size, pc, params ); break;
case 82: sub_BBB( block_size, pc, params ); break;
case 83: sub_HHH( block_size, pc, params ); break;
case 84: sub_LLL( block_size, pc, params ); break;
case 85: sub_QQQ( block_size, pc, params ); break;
case 86: sub_fff( block_size, pc, params ); break;
case 87: sub_ddd( block_size, pc, params ); break;
case 88: mult_111( block_size, pc, params ); break;
case 89: mult_bbb( block_size, pc, params ); break;
case 90: mult_hhh( block_size, pc, params ); break;
case 91: mult_lll( block_size, pc, params ); break;
case 92: mult_qqq( block_size, pc, params ); break;
case 93: mult_BBB( block_size, pc, params ); break;
case 94: mult_HHH( block_size, pc, params ); break;
case 95: mult_LLL( block_size, pc, params ); break;
case 96: mult_QQQ( block_size, pc, params ); break;
case 97: mult_fff( block_size, pc, params ); break;
case 98: mult_ddd( block_size, pc, params ); break;
case 99: div_111( block_size, pc, params ); break;
case 100: div_bbb( block_size, pc, params ); break;
case 101: div_hhh( block_size, pc, params ); break;
case 102: div_lll( block_size, pc, params ); break;
case 103: div_qqq( block_size, pc, params ); break;
case 104: div_BBB( block_size, pc, params ); break;
case 105: div_HHH( block_size, pc, params ); break;
case 106: div_LLL( block_size, pc, params ); break;
case 107: div_QQQ( block_size, pc, params ); break;
case 108: div_fff( block_size, pc, params ); break;
case 109: div_ddd( block_size, pc, params ); break;
case 110: pow_fff( block_size, pc, params ); break;
case 111: pow_ddd( block_size, pc, params ); break;
case 112: mod_111( block_size, pc, params ); break;
case 113: mod_bbb( block_size, pc, params ); break;
case 114: mod_hhh( block_size, pc, params ); break;
case 115: mod_lll( block_size, pc, params ); break;
case 116: mod_qqq( block_size, pc, params ); break;
case 117: mod_BBB( block_size, pc, params ); break;
case 118: mod_HHH( block_size, pc, params ); break;
case 119: mod_LLL( block_size, pc, params ); break;
case 120: mod_QQQ( block_size, pc, params ); break;
case 121: mod_fff( block_size, pc, params ); break;
case 122: mod_ddd( block_size, pc, params ); break;
case 123: where_1111( block_size, pc, params ); break;
case 124: where_b1bb( block_size, pc, params ); break;
case 125: where_h1hh( block_size, pc, params ); break;
case 126: where_l1ll( block_size, pc, params ); break;
case 127: where_q1qq( block_size, pc, params ); break;
case 128: where_B1BB( block_size, pc, params ); break;
case 129: where_H1HH( block_size, pc, params ); break;
case 130: where_L1LL( block_size, pc, params ); break;
case 131: where_Q1QQ( block_size, pc, params ); break;
case 132: where_f1ff( block_size, pc, params ); break;
case 133: where_d1dd( block_size, pc, params ); break;
case 134: ones_like_1( block_size, pc, params ); break;
case 135: ones_like_b( block_size, pc, params ); break;
case 136: ones_like_h( block_size, pc, params ); break;
case 137: ones_like_l( block_size, pc, params ); break;
case 138: ones_like_q( block_size, pc, params ); break;
case 139: ones_like_B( block_size, pc, params ); break;
case 140: ones_like_H( block_size, pc, params ); break;
case 141: ones_like_L( block_size, pc, params ); break;
case 142: ones_like_Q( block_size, pc, params ); break;
case 143: ones_like_f( block_size, pc, params ); break;
case 144: ones_like_d( block_size, pc, params ); break;
case 145: neg_bb( block_size, pc, params ); break;
case 146: neg_hh( block_size, pc, params ); break;
case 147: neg_ll( block_size, pc, params ); break;
case 148: neg_qq( block_size, pc, params ); break;
case 149: neg_ff( block_size, pc, params ); break;
case 150: neg_dd( block_size, pc, params ); break;
case 151: lshift_bbb( block_size, pc, params ); break;
case 152: lshift_hhh( block_size, pc, params ); break;
case 153: lshift_lll( block_size, pc, params ); break;
case 154: lshift_qqq( block_size, pc, params ); break;
case 155: lshift_BBB( block_size, pc, params ); break;
case 156: lshift_HHH( block_size, pc, params ); break;
case 157: lshift_LLL( block_size, pc, params ); break;
case 158: lshift_QQQ( block_size, pc, params ); break;
case 159: rshift_bbb( block_size, pc, params ); break;
case 160: rshift_hhh( block_size, pc, params ); break;
case 161: rshift_lll( block_size, pc, params ); break;
case 162: rshift_qqq( block_size, pc, params ); break;
case 163: rshift_BBB( block_size, pc, params ); break;
case 164: rshift_HHH( block_size, pc, params ); break;
case 165: rshift_LLL( block_size, pc, params ); break;
case 166: rshift_QQQ( block_size, pc, params ); break;
case 167: bitand_111( block_size, pc, params ); break;
case 168: bitand_bbb( block_size, pc, params ); break;
case 169: bitand_hhh( block_size, pc, params ); break;
case 170: bitand_lll( block_size, pc, params ); break;
case 171: bitand_qqq( block_size, pc, params ); break;
case 172: bitand_BBB( block_size, pc, params ); break;
case 173: bitand_HHH( block_size, pc, params ); break;
case 174: bitand_LLL( block_size, pc, params ); break;
case 175: bitand_QQQ( block_size, pc, params ); break;
case 176: bitor_111( block_size, pc, params ); break;
case 177: bitor_bbb( block_size, pc, params ); break;
case 178: bitor_hhh( block_size, pc, params ); break;
case 179: bitor_lll( block_size, pc, params ); break;
case 180: bitor_qqq( block_size, pc, params ); break;
case 181: bitor_BBB( block_size, pc, params ); break;
case 182: bitor_HHH( block_size, pc, params ); break;
case 183: bitor_LLL( block_size, pc, params ); break;
case 184: bitor_QQQ( block_size, pc, params ); break;
case 185: bitxor_111( block_size, pc, params ); break;
case 186: bitxor_bbb( block_size, pc, params ); break;
case 187: bitxor_hhh( block_size, pc, params ); break;
case 188: bitxor_lll( block_size, pc, params ); break;
case 189: bitxor_qqq( block_size, pc, params ); break;
case 190: bitxor_BBB( block_size, pc, params ); break;
case 191: bitxor_HHH( block_size, pc, params ); break;
case 192: bitxor_LLL( block_size, pc, params ); break;
case 193: bitxor_QQQ( block_size, pc, params ); break;
case 194: and_111( block_size, pc, params ); break;
case 195: or_111( block_size, pc, params ); break;
case 196: gt_111( block_size, pc, params ); break;
case 197: gt_bbb( block_size, pc, params ); break;
case 198: gt_hhh( block_size, pc, params ); break;
case 199: gt_lll( block_size, pc, params ); break;
case 200: gt_qqq( block_size, pc, params ); break;
case 201: gt_BBB( block_size, pc, params ); break;
case 202: gt_HHH( block_size, pc, params ); break;
case 203: gt_LLL( block_size, pc, params ); break;
case 204: gt_QQQ( block_size, pc, params ); break;
case 205: gt_fff( block_size, pc, params ); break;
case 206: gt_ddd( block_size, pc, params ); break;
case 207: gte_111( block_size, pc, params ); break;
case 208: gte_bbb( block_size, pc, params ); break;
case 209: gte_hhh( block_size, pc, params ); break;
case 210: gte_lll( block_size, pc, params ); break;
case 211: gte_qqq( block_size, pc, params ); break;
case 212: gte_BBB( block_size, pc, params ); break;
case 213: gte_HHH( block_size, pc, params ); break;
case 214: gte_LLL( block_size, pc, params ); break;
case 215: gte_QQQ( block_size, pc, params ); break;
case 216: gte_fff( block_size, pc, params ); break;
case 217: gte_ddd( block_size, pc, params ); break;
case 218: lt_111( block_size, pc, params ); break;
case 219: lt_bbb( block_size, pc, params ); break;
case 220: lt_hhh( block_size, pc, params ); break;
case 221: lt_lll( block_size, pc, params ); break;
case 222: lt_qqq( block_size, pc, params ); break;
case 223: lt_BBB( block_size, pc, params ); break;
case 224: lt_HHH( block_size, pc, params ); break;
case 225: lt_LLL( block_size, pc, params ); break;
case 226: lt_QQQ( block_size, pc, params ); break;
case 227: lt_fff( block_size, pc, params ); break;
case 228: lt_ddd( block_size, pc, params ); break;
case 229: lte_111( block_size, pc, params ); break;
case 230: lte_bbb( block_size, pc, params ); break;
case 231: lte_hhh( block_size, pc, params ); break;
case 232: lte_lll( block_size, pc, params ); break;
case 233: lte_qqq( block_size, pc, params ); break;
case 234: lte_BBB( block_size, pc, params ); break;
case 235: lte_HHH( block_size, pc, params ); break;
case 236: lte_LLL( block_size, pc, params ); break;
case 237: lte_QQQ( block_size, pc, params ); break;
case 238: lte_fff( block_size, pc, params ); break;
case 239: lte_ddd( block_size, pc, params ); break;
case 240: eq_111( block_size, pc, params ); break;
case 241: eq_bbb( block_size, pc, params ); break;
case 242: eq_hhh( block_size, pc, params ); break;
case 243: eq_lll( block_size, pc, params ); break;
case 244: eq_qqq( block_size, pc, params ); break;
case 245: eq_BBB( block_size, pc, params ); break;
case 246: eq_HHH( block_size, pc, params ); break;
case 247: eq_LLL( block_size, pc, params ); break;
case 248: eq_QQQ( block_size, pc, params ); break;
case 249: eq_fff( block_size, pc, params ); break;
case 250: eq_ddd( block_size, pc, params ); break;
case 251: noteq_111( block_size, pc, params ); break;
case 252: noteq_bbb( block_size, pc, params ); break;
case 253: noteq_hhh( block_size, pc, params ); break;
case 254: noteq_lll( block_size, pc, params ); break;
case 255: noteq_qqq( block_size, pc, params ); break;
case 256: noteq_BBB( block_size, pc, params ); break;
case 257: noteq_HHH( block_size, pc, params ); break;
case 258: noteq_LLL( block_size, pc, params ); break;
case 259: noteq_QQQ( block_size, pc, params ); break;
case 260: noteq_fff( block_size, pc, params ); break;
case 261: noteq_ddd( block_size, pc, params ); break;
case 262: abs_ff( block_size, pc, params ); break;
case 263: abs_dd( block_size, pc, params ); break;
case 264: arccos_ff( block_size, pc, params ); break;
case 265: arccos_dd( block_size, pc, params ); break;
case 266: arcsin_ff( block_size, pc, params ); break;
case 267: arcsin_dd( block_size, pc, params ); break;
case 268: arctan_ff( block_size, pc, params ); break;
case 269: arctan_dd( block_size, pc, params ); break;
case 270: arctan2_fff( block_size, pc, params ); break;
case 271: arctan2_ddd( block_size, pc, params ); break;
case 272: ceil_ff( block_size, pc, params ); break;
case 273: ceil_dd( block_size, pc, params ); break;
case 274: cos_ff( block_size, pc, params ); break;
case 275: cos_dd( block_size, pc, params ); break;
case 276: cosh_ff( block_size, pc, params ); break;
case 277: cosh_dd( block_size, pc, params ); break;
case 278: exp_ff( block_size, pc, params ); break;
case 279: exp_dd( block_size, pc, params ); break;
case 280: fabs_ff( block_size, pc, params ); break;
case 281: fabs_dd( block_size, pc, params ); break;
case 282: floor_ff( block_size, pc, params ); break;
case 283: floor_dd( block_size, pc, params ); break;
case 284: fmod_fff( block_size, pc, params ); break;
case 285: fmod_ddd( block_size, pc, params ); break;
case 286: log_ff( block_size, pc, params ); break;
case 287: log_dd( block_size, pc, params ); break;
case 288: log10_ff( block_size, pc, params ); break;
case 289: log10_dd( block_size, pc, params ); break;
case 290: power_fff( block_size, pc, params ); break;
case 291: power_ddd( block_size, pc, params ); break;
case 292: power_ffl( block_size, pc, params ); break;
case 293: power_ddl( block_size, pc, params ); break;
case 294: sin_ff( block_size, pc, params ); break;
case 295: sin_dd( block_size, pc, params ); break;
case 296: sinh_ff( block_size, pc, params ); break;
case 297: sinh_dd( block_size, pc, params ); break;
case 298: sqrt_ff( block_size, pc, params ); break;
case 299: sqrt_dd( block_size, pc, params ); break;
case 300: tan_ff( block_size, pc, params ); break;
case 301: tan_dd( block_size, pc, params ); break;
case 302: tanh_ff( block_size, pc, params ); break;
case 303: tanh_dd( block_size, pc, params ); break;
case 304: fpclassify_lf( block_size, pc, params ); break;
case 305: fpclassify_ld( block_size, pc, params ); break;
case 306: isfinite_1f( block_size, pc, params ); break;
case 307: isfinite_1d( block_size, pc, params ); break;
case 308: isinf_1f( block_size, pc, params ); break;
case 309: isinf_1d( block_size, pc, params ); break;
case 310: isnan_1f( block_size, pc, params ); break;
case 311: isnan_1d( block_size, pc, params ); break;
case 312: isnormal_1f( block_size, pc, params ); break;
case 313: isnormal_1d( block_size, pc, params ); break;
case 314: signbit_1f( block_size, pc, params ); break;
case 315: signbit_1d( block_size, pc, params ); break;
case 316: arccosh_ff( block_size, pc, params ); break;
case 317: arccosh_dd( block_size, pc, params ); break;
case 318: arcsinh_ff( block_size, pc, params ); break;
case 319: arcsinh_dd( block_size, pc, params ); break;
case 320: arctanh_ff( block_size, pc, params ); break;
case 321: arctanh_dd( block_size, pc, params ); break;
case 322: cbrt_ff( block_size, pc, params ); break;
case 323: cbrt_dd( block_size, pc, params ); break;
case 324: copysign_fff( block_size, pc, params ); break;
case 325: copysign_ddd( block_size, pc, params ); break;
case 326: erf_ff( block_size, pc, params ); break;
case 327: erf_dd( block_size, pc, params ); break;
case 328: erfc_ff( block_size, pc, params ); break;
case 329: erfc_dd( block_size, pc, params ); break;
case 330: exp2_ff( block_size, pc, params ); break;
case 331: exp2_dd( block_size, pc, params ); break;
case 332: expm1_ff( block_size, pc, params ); break;
case 333: expm1_dd( block_size, pc, params ); break;
case 334: fdim_fff( block_size, pc, params ); break;
case 335: fdim_ddd( block_size, pc, params ); break;
case 336: fma_ffff( block_size, pc, params ); break;
case 337: fma_dddd( block_size, pc, params ); break;
case 338: fmax_fff( block_size, pc, params ); break;
case 339: fmax_ddd( block_size, pc, params ); break;
case 340: fmin_fff( block_size, pc, params ); break;
case 341: fmin_ddd( block_size, pc, params ); break;
case 342: hypot_fff( block_size, pc, params ); break;
case 343: hypot_ddd( block_size, pc, params ); break;
case 344: ilogb_lf( block_size, pc, params ); break;
case 345: ilogb_ld( block_size, pc, params ); break;
case 346: lgamma_ff( block_size, pc, params ); break;
case 347: lgamma_dd( block_size, pc, params ); break;
case 348: log1p_ff( block_size, pc, params ); break;
case 349: log1p_dd( block_size, pc, params ); break;
case 350: log2_ff( block_size, pc, params ); break;
case 351: log2_dd( block_size, pc, params ); break;
case 352: logb_ff( block_size, pc, params ); break;
case 353: logb_dd( block_size, pc, params ); break;
case 354: lrint_qf( block_size, pc, params ); break;
case 355: lrint_qd( block_size, pc, params ); break;
case 356: lround_qf( block_size, pc, params ); break;
case 357: lround_qd( block_size, pc, params ); break;
case 358: nearbyint_qf( block_size, pc, params ); break;
case 359: nearbyint_qd( block_size, pc, params ); break;
case 360: nextafter_fff( block_size, pc, params ); break;
case 361: nextafter_ddd( block_size, pc, params ); break;
case 362: nexttoward_fff( block_size, pc, params ); break;
case 363: nexttoward_ddd( block_size, pc, params ); break;
case 364: remainder_fff( block_size, pc, params ); break;
case 365: remainder_ddd( block_size, pc, params ); break;
case 366: rint_ff( block_size, pc, params ); break;
case 367: rint_dd( block_size, pc, params ); break;
case 368: round_lf( block_size, pc, params ); break;
case 369: round_ld( block_size, pc, params ); break;
case 370: scalbln_ffq( block_size, pc, params ); break;
case 371: scalbln_ddq( block_size, pc, params ); break;
case 372: tgamma_ff( block_size, pc, params ); break;
case 373: tgamma_dd( block_size, pc, params ); break;
case 374: trunc_ff( block_size, pc, params ); break;
case 375: trunc_dd( block_size, pc, params ); break;
case 376: abs_fF( block_size, pc, params ); break;
case 377: abs_dD( block_size, pc, params ); break;
case 378: add_FFF( block_size, pc, params ); break;
case 379: add_DDD( block_size, pc, params ); break;
case 380: sub_FFF( block_size, pc, params ); break;
case 381: sub_DDD( block_size, pc, params ); break;
case 382: mult_FFF( block_size, pc, params ); break;
case 383: mult_DDD( block_size, pc, params ); break;
case 384: div_FFF( block_size, pc, params ); break;
case 385: div_DDD( block_size, pc, params ); break;
case 386: neg_FF( block_size, pc, params ); break;
case 387: neg_DD( block_size, pc, params ); break;
case 388: conj_FF( block_size, pc, params ); break;
case 389: conj_DD( block_size, pc, params ); break;
case 390: conj_ff( block_size, pc, params ); break;
case 391: conj_dd( block_size, pc, params ); break;
case 392: sqrt_FF( block_size, pc, params ); break;
case 393: sqrt_DD( block_size, pc, params ); break;
case 394: log_FF( block_size, pc, params ); break;
case 395: log_DD( block_size, pc, params ); break;
case 396: log1p_FF( block_size, pc, params ); break;
case 397: log1p_DD( block_size, pc, params ); break;
case 398: log10_FF( block_size, pc, params ); break;
case 399: log10_DD( block_size, pc, params ); break;
case 400: exp_FF( block_size, pc, params ); break;
case 401: exp_DD( block_size, pc, params ); break;
case 402: expm1_FF( block_size, pc, params ); break;
case 403: expm1_DD( block_size, pc, params ); break;
case 404: pow_FFF( block_size, pc, params ); break;
case 405: pow_DDD( block_size, pc, params ); break;
case 406: arccos_FF( block_size, pc, params ); break;
case 407: arccos_DD( block_size, pc, params ); break;
case 408: arccosh_FF( block_size, pc, params ); break;
case 409: arccosh_DD( block_size, pc, params ); break;
case 410: arcsin_FF( block_size, pc, params ); break;
case 411: arcsin_DD( block_size, pc, params ); break;
case 412: arcsinh_FF( block_size, pc, params ); break;
case 413: arcsinh_DD( block_size, pc, params ); break;
case 414: arctan_FF( block_size, pc, params ); break;
case 415: arctan_DD( block_size, pc, params ); break;
case 416: arctanh_FF( block_size, pc, params ); break;
case 417: arctanh_DD( block_size, pc, params ); break;
case 418: cos_FF( block_size, pc, params ); break;
case 419: cos_DD( block_size, pc, params ); break;
case 420: cosh_FF( block_size, pc, params ); break;
case 421: cosh_DD( block_size, pc, params ); break;
case 422: sin_FF( block_size, pc, params ); break;
case 423: sin_DD( block_size, pc, params ); break;
case 424: sinh_FF( block_size, pc, params ); break;
case 425: sinh_DD( block_size, pc, params ); break;
case 426: tan_FF( block_size, pc, params ); break;
case 427: tan_DD( block_size, pc, params ); break;
case 428: tanh_FF( block_size, pc, params ); break;
case 429: tanh_DD( block_size, pc, params ); break;
// End of GENERATED CODE BLOCK

        default:
            //*pc_error = pc;
            return -3;
            break;
        }
    }

#ifndef NO_OUTPUT_BUFFERING
    // If output buffering was necessary, copy the buffer to the output
    //printf( "TODO: output buffering disabled.\n" );
//    if(params->outBuffer != NULL) {
//        memcpy(iterDataPtr[0], params->outBuffer, GET_RETURN_REG(params).itemsize * BLOCK_SIZE);
//    }
#endif // NO_OUTPUT_BUFFERING
}


