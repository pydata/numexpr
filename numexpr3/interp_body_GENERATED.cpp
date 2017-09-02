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
        if( params->registers[J].kind == KIND_ARRAY || params->registers[J].kind == KIND_RETURN ) {
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

        // Sample debug output, to be pasted into functions_GENERATED.cpp 
        // as needed: 
        // printf( "Arg1(%d@%p) + ARG2(%d@%p) => DEST(%d@%p)\n", arg1, x1, arg2, x2, store_in, dest );
        
        // TODO: BLOCK_SIZE1 is the number of operations, not the array block size,
        // so the memory block scales with itemsize...
        // printf( "Exec op: %d\n", params->program[pc].op );

        switch (params->program[pc].op) {
        case 0: 
            break;
        case 1: cast_b1( task_size, pc, params ); break;
case 2: cast_h1( task_size, pc, params ); break;
case 3: cast_l1( task_size, pc, params ); break;
case 4: cast_q1( task_size, pc, params ); break;
case 5: cast_B1( task_size, pc, params ); break;
case 6: cast_H1( task_size, pc, params ); break;
case 7: cast_L1( task_size, pc, params ); break;
case 8: cast_Q1( task_size, pc, params ); break;
case 9: cast_f1( task_size, pc, params ); break;
case 10: cast_d1( task_size, pc, params ); break;
case 11: cast_F1( task_size, pc, params ); break;
case 12: cast_D1( task_size, pc, params ); break;
case 13: cast_hb( task_size, pc, params ); break;
case 14: cast_lb( task_size, pc, params ); break;
case 15: cast_qb( task_size, pc, params ); break;
case 16: cast_fb( task_size, pc, params ); break;
case 17: cast_db( task_size, pc, params ); break;
case 18: cast_Fb( task_size, pc, params ); break;
case 19: cast_Db( task_size, pc, params ); break;
case 20: cast_lh( task_size, pc, params ); break;
case 21: cast_qh( task_size, pc, params ); break;
case 22: cast_fh( task_size, pc, params ); break;
case 23: cast_dh( task_size, pc, params ); break;
case 24: cast_Fh( task_size, pc, params ); break;
case 25: cast_Dh( task_size, pc, params ); break;
case 26: cast_ql( task_size, pc, params ); break;
case 27: cast_dl( task_size, pc, params ); break;
case 28: cast_Dl( task_size, pc, params ); break;
case 29: cast_dq( task_size, pc, params ); break;
case 30: cast_Dq( task_size, pc, params ); break;
case 31: cast_hB( task_size, pc, params ); break;
case 32: cast_lB( task_size, pc, params ); break;
case 33: cast_qB( task_size, pc, params ); break;
case 34: cast_HB( task_size, pc, params ); break;
case 35: cast_LB( task_size, pc, params ); break;
case 36: cast_QB( task_size, pc, params ); break;
case 37: cast_fB( task_size, pc, params ); break;
case 38: cast_dB( task_size, pc, params ); break;
case 39: cast_FB( task_size, pc, params ); break;
case 40: cast_DB( task_size, pc, params ); break;
case 41: cast_lH( task_size, pc, params ); break;
case 42: cast_qH( task_size, pc, params ); break;
case 43: cast_LH( task_size, pc, params ); break;
case 44: cast_QH( task_size, pc, params ); break;
case 45: cast_fH( task_size, pc, params ); break;
case 46: cast_dH( task_size, pc, params ); break;
case 47: cast_FH( task_size, pc, params ); break;
case 48: cast_DH( task_size, pc, params ); break;
case 49: cast_qL( task_size, pc, params ); break;
case 50: cast_QL( task_size, pc, params ); break;
case 51: cast_dL( task_size, pc, params ); break;
case 52: cast_DL( task_size, pc, params ); break;
case 53: cast_dQ( task_size, pc, params ); break;
case 54: cast_DQ( task_size, pc, params ); break;
case 55: cast_df( task_size, pc, params ); break;
case 56: cast_Ff( task_size, pc, params ); break;
case 57: cast_Df( task_size, pc, params ); break;
case 58: cast_Dd( task_size, pc, params ); break;
case 59: cast_DF( task_size, pc, params ); break;
case 60: copy_11( task_size, pc, params ); break;
case 61: copy_bb( task_size, pc, params ); break;
case 62: copy_hh( task_size, pc, params ); break;
case 63: copy_ll( task_size, pc, params ); break;
case 64: copy_qq( task_size, pc, params ); break;
case 65: copy_BB( task_size, pc, params ); break;
case 66: copy_HH( task_size, pc, params ); break;
case 67: copy_LL( task_size, pc, params ); break;
case 68: copy_QQ( task_size, pc, params ); break;
case 69: copy_ff( task_size, pc, params ); break;
case 70: copy_dd( task_size, pc, params ); break;
case 71: copy_FF( task_size, pc, params ); break;
case 72: copy_DD( task_size, pc, params ); break;
case 73: add_111( task_size, pc, params ); break;
case 74: add_bbb( task_size, pc, params ); break;
case 75: add_hhh( task_size, pc, params ); break;
case 76: add_lll( task_size, pc, params ); break;
case 77: add_qqq( task_size, pc, params ); break;
case 78: add_BBB( task_size, pc, params ); break;
case 79: add_HHH( task_size, pc, params ); break;
case 80: add_LLL( task_size, pc, params ); break;
case 81: add_QQQ( task_size, pc, params ); break;
case 82: add_fff( task_size, pc, params ); break;
case 83: add_ddd( task_size, pc, params ); break;
case 84: sub_bbb( task_size, pc, params ); break;
case 85: sub_hhh( task_size, pc, params ); break;
case 86: sub_lll( task_size, pc, params ); break;
case 87: sub_qqq( task_size, pc, params ); break;
case 88: sub_BBB( task_size, pc, params ); break;
case 89: sub_HHH( task_size, pc, params ); break;
case 90: sub_LLL( task_size, pc, params ); break;
case 91: sub_QQQ( task_size, pc, params ); break;
case 92: sub_fff( task_size, pc, params ); break;
case 93: sub_ddd( task_size, pc, params ); break;
case 94: mult_111( task_size, pc, params ); break;
case 95: mult_bbb( task_size, pc, params ); break;
case 96: mult_hhh( task_size, pc, params ); break;
case 97: mult_lll( task_size, pc, params ); break;
case 98: mult_qqq( task_size, pc, params ); break;
case 99: mult_BBB( task_size, pc, params ); break;
case 100: mult_HHH( task_size, pc, params ); break;
case 101: mult_LLL( task_size, pc, params ); break;
case 102: mult_QQQ( task_size, pc, params ); break;
case 103: mult_fff( task_size, pc, params ); break;
case 104: mult_ddd( task_size, pc, params ); break;
case 105: div_d11( task_size, pc, params ); break;
case 106: div_dbb( task_size, pc, params ); break;
case 107: div_dhh( task_size, pc, params ); break;
case 108: div_dll( task_size, pc, params ); break;
case 109: div_dqq( task_size, pc, params ); break;
case 110: div_dBB( task_size, pc, params ); break;
case 111: div_dHH( task_size, pc, params ); break;
case 112: div_dLL( task_size, pc, params ); break;
case 113: div_dQQ( task_size, pc, params ); break;
case 114: div_fff( task_size, pc, params ); break;
case 115: div_ddd( task_size, pc, params ); break;
case 116: floordiv_bbb( task_size, pc, params ); break;
case 117: floordiv_hhh( task_size, pc, params ); break;
case 118: floordiv_lll( task_size, pc, params ); break;
case 119: floordiv_qqq( task_size, pc, params ); break;
case 120: floordiv_BBB( task_size, pc, params ); break;
case 121: floordiv_HHH( task_size, pc, params ); break;
case 122: floordiv_LLL( task_size, pc, params ); break;
case 123: floordiv_QQQ( task_size, pc, params ); break;
case 124: pow_fff( task_size, pc, params ); break;
case 125: pow_ddd( task_size, pc, params ); break;
case 126: mod_fff( task_size, pc, params ); break;
case 127: mod_ddd( task_size, pc, params ); break;
case 128: where_1111( task_size, pc, params ); break;
case 129: where_b1bb( task_size, pc, params ); break;
case 130: where_h1hh( task_size, pc, params ); break;
case 131: where_l1ll( task_size, pc, params ); break;
case 132: where_q1qq( task_size, pc, params ); break;
case 133: where_B1BB( task_size, pc, params ); break;
case 134: where_H1HH( task_size, pc, params ); break;
case 135: where_L1LL( task_size, pc, params ); break;
case 136: where_Q1QQ( task_size, pc, params ); break;
case 137: where_f1ff( task_size, pc, params ); break;
case 138: where_d1dd( task_size, pc, params ); break;
case 139: where_F1FF( task_size, pc, params ); break;
case 140: where_D1DD( task_size, pc, params ); break;
case 141: ones_like_11( task_size, pc, params ); break;
case 142: ones_like_bb( task_size, pc, params ); break;
case 143: ones_like_hh( task_size, pc, params ); break;
case 144: ones_like_ll( task_size, pc, params ); break;
case 145: ones_like_qq( task_size, pc, params ); break;
case 146: ones_like_BB( task_size, pc, params ); break;
case 147: ones_like_HH( task_size, pc, params ); break;
case 148: ones_like_LL( task_size, pc, params ); break;
case 149: ones_like_QQ( task_size, pc, params ); break;
case 150: ones_like_ff( task_size, pc, params ); break;
case 151: ones_like_dd( task_size, pc, params ); break;
case 152: usub_bb( task_size, pc, params ); break;
case 153: usub_hh( task_size, pc, params ); break;
case 154: usub_ll( task_size, pc, params ); break;
case 155: usub_qq( task_size, pc, params ); break;
case 156: usub_ff( task_size, pc, params ); break;
case 157: usub_dd( task_size, pc, params ); break;
case 158: lshift_bbb( task_size, pc, params ); break;
case 159: lshift_hhh( task_size, pc, params ); break;
case 160: lshift_lll( task_size, pc, params ); break;
case 161: lshift_qqq( task_size, pc, params ); break;
case 162: lshift_BBB( task_size, pc, params ); break;
case 163: lshift_HHH( task_size, pc, params ); break;
case 164: lshift_LLL( task_size, pc, params ); break;
case 165: lshift_QQQ( task_size, pc, params ); break;
case 166: rshift_bbb( task_size, pc, params ); break;
case 167: rshift_hhh( task_size, pc, params ); break;
case 168: rshift_lll( task_size, pc, params ); break;
case 169: rshift_qqq( task_size, pc, params ); break;
case 170: rshift_BBB( task_size, pc, params ); break;
case 171: rshift_HHH( task_size, pc, params ); break;
case 172: rshift_LLL( task_size, pc, params ); break;
case 173: rshift_QQQ( task_size, pc, params ); break;
case 174: bitand_111( task_size, pc, params ); break;
case 175: bitand_bbb( task_size, pc, params ); break;
case 176: bitand_hhh( task_size, pc, params ); break;
case 177: bitand_lll( task_size, pc, params ); break;
case 178: bitand_qqq( task_size, pc, params ); break;
case 179: bitand_BBB( task_size, pc, params ); break;
case 180: bitand_HHH( task_size, pc, params ); break;
case 181: bitand_LLL( task_size, pc, params ); break;
case 182: bitand_QQQ( task_size, pc, params ); break;
case 183: bitor_111( task_size, pc, params ); break;
case 184: bitor_bbb( task_size, pc, params ); break;
case 185: bitor_hhh( task_size, pc, params ); break;
case 186: bitor_lll( task_size, pc, params ); break;
case 187: bitor_qqq( task_size, pc, params ); break;
case 188: bitor_BBB( task_size, pc, params ); break;
case 189: bitor_HHH( task_size, pc, params ); break;
case 190: bitor_LLL( task_size, pc, params ); break;
case 191: bitor_QQQ( task_size, pc, params ); break;
case 192: bitxor_111( task_size, pc, params ); break;
case 193: bitxor_bbb( task_size, pc, params ); break;
case 194: bitxor_hhh( task_size, pc, params ); break;
case 195: bitxor_lll( task_size, pc, params ); break;
case 196: bitxor_qqq( task_size, pc, params ); break;
case 197: bitxor_BBB( task_size, pc, params ); break;
case 198: bitxor_HHH( task_size, pc, params ); break;
case 199: bitxor_LLL( task_size, pc, params ); break;
case 200: bitxor_QQQ( task_size, pc, params ); break;
case 201: logical_and_111( task_size, pc, params ); break;
case 202: logical_or_111( task_size, pc, params ); break;
case 203: gt_111( task_size, pc, params ); break;
case 204: gt_1bb( task_size, pc, params ); break;
case 205: gt_1hh( task_size, pc, params ); break;
case 206: gt_1ll( task_size, pc, params ); break;
case 207: gt_1qq( task_size, pc, params ); break;
case 208: gt_1BB( task_size, pc, params ); break;
case 209: gt_1HH( task_size, pc, params ); break;
case 210: gt_1LL( task_size, pc, params ); break;
case 211: gt_1QQ( task_size, pc, params ); break;
case 212: gt_1ff( task_size, pc, params ); break;
case 213: gt_1dd( task_size, pc, params ); break;
case 214: gte_111( task_size, pc, params ); break;
case 215: gte_1bb( task_size, pc, params ); break;
case 216: gte_1hh( task_size, pc, params ); break;
case 217: gte_1ll( task_size, pc, params ); break;
case 218: gte_1qq( task_size, pc, params ); break;
case 219: gte_1BB( task_size, pc, params ); break;
case 220: gte_1HH( task_size, pc, params ); break;
case 221: gte_1LL( task_size, pc, params ); break;
case 222: gte_1QQ( task_size, pc, params ); break;
case 223: gte_1ff( task_size, pc, params ); break;
case 224: gte_1dd( task_size, pc, params ); break;
case 225: lt_111( task_size, pc, params ); break;
case 226: lt_1bb( task_size, pc, params ); break;
case 227: lt_1hh( task_size, pc, params ); break;
case 228: lt_1ll( task_size, pc, params ); break;
case 229: lt_1qq( task_size, pc, params ); break;
case 230: lt_1BB( task_size, pc, params ); break;
case 231: lt_1HH( task_size, pc, params ); break;
case 232: lt_1LL( task_size, pc, params ); break;
case 233: lt_1QQ( task_size, pc, params ); break;
case 234: lt_1ff( task_size, pc, params ); break;
case 235: lt_1dd( task_size, pc, params ); break;
case 236: lte_111( task_size, pc, params ); break;
case 237: lte_1bb( task_size, pc, params ); break;
case 238: lte_1hh( task_size, pc, params ); break;
case 239: lte_1ll( task_size, pc, params ); break;
case 240: lte_1qq( task_size, pc, params ); break;
case 241: lte_1BB( task_size, pc, params ); break;
case 242: lte_1HH( task_size, pc, params ); break;
case 243: lte_1LL( task_size, pc, params ); break;
case 244: lte_1QQ( task_size, pc, params ); break;
case 245: lte_1ff( task_size, pc, params ); break;
case 246: lte_1dd( task_size, pc, params ); break;
case 247: eq_111( task_size, pc, params ); break;
case 248: eq_1bb( task_size, pc, params ); break;
case 249: eq_1hh( task_size, pc, params ); break;
case 250: eq_1ll( task_size, pc, params ); break;
case 251: eq_1qq( task_size, pc, params ); break;
case 252: eq_1BB( task_size, pc, params ); break;
case 253: eq_1HH( task_size, pc, params ); break;
case 254: eq_1LL( task_size, pc, params ); break;
case 255: eq_1QQ( task_size, pc, params ); break;
case 256: eq_1ff( task_size, pc, params ); break;
case 257: eq_1dd( task_size, pc, params ); break;
case 258: noteq_111( task_size, pc, params ); break;
case 259: noteq_1bb( task_size, pc, params ); break;
case 260: noteq_1hh( task_size, pc, params ); break;
case 261: noteq_1ll( task_size, pc, params ); break;
case 262: noteq_1qq( task_size, pc, params ); break;
case 263: noteq_1BB( task_size, pc, params ); break;
case 264: noteq_1HH( task_size, pc, params ); break;
case 265: noteq_1LL( task_size, pc, params ); break;
case 266: noteq_1QQ( task_size, pc, params ); break;
case 267: noteq_1ff( task_size, pc, params ); break;
case 268: noteq_1dd( task_size, pc, params ); break;
case 269: abs_bb( task_size, pc, params ); break;
case 270: abs_hh( task_size, pc, params ); break;
case 271: abs_ll( task_size, pc, params ); break;
case 272: abs_qq( task_size, pc, params ); break;
case 273: abs_ff( task_size, pc, params ); break;
case 274: abs_dd( task_size, pc, params ); break;
case 275: arccos_ff( task_size, pc, params ); break;
case 276: arccos_dd( task_size, pc, params ); break;
case 277: arcsin_ff( task_size, pc, params ); break;
case 278: arcsin_dd( task_size, pc, params ); break;
case 279: arctan_ff( task_size, pc, params ); break;
case 280: arctan_dd( task_size, pc, params ); break;
case 281: arctan2_fff( task_size, pc, params ); break;
case 282: arctan2_ddd( task_size, pc, params ); break;
case 283: ceil_ff( task_size, pc, params ); break;
case 284: ceil_dd( task_size, pc, params ); break;
case 285: cos_ff( task_size, pc, params ); break;
case 286: cos_dd( task_size, pc, params ); break;
case 287: cosh_ff( task_size, pc, params ); break;
case 288: cosh_dd( task_size, pc, params ); break;
case 289: exp_ff( task_size, pc, params ); break;
case 290: exp_dd( task_size, pc, params ); break;
case 291: fabs_ff( task_size, pc, params ); break;
case 292: fabs_dd( task_size, pc, params ); break;
case 293: floor_ff( task_size, pc, params ); break;
case 294: floor_dd( task_size, pc, params ); break;
case 295: fmod_fff( task_size, pc, params ); break;
case 296: fmod_ddd( task_size, pc, params ); break;
case 297: log_ff( task_size, pc, params ); break;
case 298: log_dd( task_size, pc, params ); break;
case 299: log10_ff( task_size, pc, params ); break;
case 300: log10_dd( task_size, pc, params ); break;
case 301: sin_ff( task_size, pc, params ); break;
case 302: sin_dd( task_size, pc, params ); break;
case 303: sinh_ff( task_size, pc, params ); break;
case 304: sinh_dd( task_size, pc, params ); break;
case 305: sqrt_ff( task_size, pc, params ); break;
case 306: sqrt_dd( task_size, pc, params ); break;
case 307: tan_ff( task_size, pc, params ); break;
case 308: tan_dd( task_size, pc, params ); break;
case 309: tanh_ff( task_size, pc, params ); break;
case 310: tanh_dd( task_size, pc, params ); break;
case 311: fpclassify_lf( task_size, pc, params ); break;
case 312: fpclassify_ld( task_size, pc, params ); break;
case 313: isfinite_1f( task_size, pc, params ); break;
case 314: isfinite_1d( task_size, pc, params ); break;
case 315: isinf_1f( task_size, pc, params ); break;
case 316: isinf_1d( task_size, pc, params ); break;
case 317: isnan_1f( task_size, pc, params ); break;
case 318: isnan_1d( task_size, pc, params ); break;
case 319: isnormal_1f( task_size, pc, params ); break;
case 320: isnormal_1d( task_size, pc, params ); break;
case 321: signbit_1f( task_size, pc, params ); break;
case 322: signbit_1d( task_size, pc, params ); break;
case 323: arccosh_ff( task_size, pc, params ); break;
case 324: arccosh_dd( task_size, pc, params ); break;
case 325: arcsinh_ff( task_size, pc, params ); break;
case 326: arcsinh_dd( task_size, pc, params ); break;
case 327: arctanh_ff( task_size, pc, params ); break;
case 328: arctanh_dd( task_size, pc, params ); break;
case 329: cbrt_ff( task_size, pc, params ); break;
case 330: cbrt_dd( task_size, pc, params ); break;
case 331: copysign_fff( task_size, pc, params ); break;
case 332: copysign_ddd( task_size, pc, params ); break;
case 333: erf_ff( task_size, pc, params ); break;
case 334: erf_dd( task_size, pc, params ); break;
case 335: erfc_ff( task_size, pc, params ); break;
case 336: erfc_dd( task_size, pc, params ); break;
case 337: exp2_ff( task_size, pc, params ); break;
case 338: exp2_dd( task_size, pc, params ); break;
case 339: expm1_ff( task_size, pc, params ); break;
case 340: expm1_dd( task_size, pc, params ); break;
case 341: fdim_fff( task_size, pc, params ); break;
case 342: fdim_ddd( task_size, pc, params ); break;
case 343: fma_ffff( task_size, pc, params ); break;
case 344: fma_dddd( task_size, pc, params ); break;
case 345: fmax_fff( task_size, pc, params ); break;
case 346: fmax_ddd( task_size, pc, params ); break;
case 347: fmin_fff( task_size, pc, params ); break;
case 348: fmin_ddd( task_size, pc, params ); break;
case 349: hypot_fff( task_size, pc, params ); break;
case 350: hypot_ddd( task_size, pc, params ); break;
case 351: ilogb_lf( task_size, pc, params ); break;
case 352: ilogb_ld( task_size, pc, params ); break;
case 353: lgamma_ff( task_size, pc, params ); break;
case 354: lgamma_dd( task_size, pc, params ); break;
case 355: log1p_ff( task_size, pc, params ); break;
case 356: log1p_dd( task_size, pc, params ); break;
case 357: log2_ff( task_size, pc, params ); break;
case 358: log2_dd( task_size, pc, params ); break;
case 359: logb_ff( task_size, pc, params ); break;
case 360: logb_dd( task_size, pc, params ); break;
case 361: lrint_qf( task_size, pc, params ); break;
case 362: lrint_qd( task_size, pc, params ); break;
case 363: lround_qf( task_size, pc, params ); break;
case 364: lround_qd( task_size, pc, params ); break;
case 365: nearbyint_qf( task_size, pc, params ); break;
case 366: nearbyint_qd( task_size, pc, params ); break;
case 367: nextafter_fff( task_size, pc, params ); break;
case 368: nextafter_ddd( task_size, pc, params ); break;
case 369: nexttoward_fff( task_size, pc, params ); break;
case 370: nexttoward_ddd( task_size, pc, params ); break;
case 371: rint_ff( task_size, pc, params ); break;
case 372: rint_dd( task_size, pc, params ); break;
case 373: round_lf( task_size, pc, params ); break;
case 374: round_ld( task_size, pc, params ); break;
case 375: scalbln_ffq( task_size, pc, params ); break;
case 376: scalbln_ddq( task_size, pc, params ); break;
case 377: tgamma_ff( task_size, pc, params ); break;
case 378: tgamma_dd( task_size, pc, params ); break;
case 379: trunc_ff( task_size, pc, params ); break;
case 380: trunc_dd( task_size, pc, params ); break;
case 381: complex_Fff( task_size, pc, params ); break;
case 382: complex_Ddd( task_size, pc, params ); break;
case 383: real_fF( task_size, pc, params ); break;
case 384: real_dD( task_size, pc, params ); break;
case 385: imag_fF( task_size, pc, params ); break;
case 386: imag_dD( task_size, pc, params ); break;
case 387: abs_fF( task_size, pc, params ); break;
case 388: abs_dD( task_size, pc, params ); break;
case 389: abs2_fF( task_size, pc, params ); break;
case 390: abs2_dD( task_size, pc, params ); break;
case 391: add_FFF( task_size, pc, params ); break;
case 392: add_DDD( task_size, pc, params ); break;
case 393: sub_FFF( task_size, pc, params ); break;
case 394: sub_DDD( task_size, pc, params ); break;
case 395: mult_FFF( task_size, pc, params ); break;
case 396: mult_DDD( task_size, pc, params ); break;
case 397: div_FFF( task_size, pc, params ); break;
case 398: div_DDD( task_size, pc, params ); break;
case 399: usub_FF( task_size, pc, params ); break;
case 400: usub_DD( task_size, pc, params ); break;
case 401: neg_FF( task_size, pc, params ); break;
case 402: neg_DD( task_size, pc, params ); break;
case 403: conj_FF( task_size, pc, params ); break;
case 404: conj_DD( task_size, pc, params ); break;
case 405: conj_ff( task_size, pc, params ); break;
case 406: conj_dd( task_size, pc, params ); break;
case 407: sqrt_FF( task_size, pc, params ); break;
case 408: sqrt_DD( task_size, pc, params ); break;
case 409: log_FF( task_size, pc, params ); break;
case 410: log_DD( task_size, pc, params ); break;
case 411: log1p_FF( task_size, pc, params ); break;
case 412: log1p_DD( task_size, pc, params ); break;
case 413: log10_FF( task_size, pc, params ); break;
case 414: log10_DD( task_size, pc, params ); break;
case 415: exp_FF( task_size, pc, params ); break;
case 416: exp_DD( task_size, pc, params ); break;
case 417: expm1_FF( task_size, pc, params ); break;
case 418: expm1_DD( task_size, pc, params ); break;
case 419: pow_FFF( task_size, pc, params ); break;
case 420: pow_DDD( task_size, pc, params ); break;
case 421: arccos_FF( task_size, pc, params ); break;
case 422: arccos_DD( task_size, pc, params ); break;
case 423: arccosh_FF( task_size, pc, params ); break;
case 424: arccosh_DD( task_size, pc, params ); break;
case 425: arcsin_FF( task_size, pc, params ); break;
case 426: arcsin_DD( task_size, pc, params ); break;
case 427: arcsinh_FF( task_size, pc, params ); break;
case 428: arcsinh_DD( task_size, pc, params ); break;
case 429: arctan_FF( task_size, pc, params ); break;
case 430: arctan_DD( task_size, pc, params ); break;
case 431: arctanh_FF( task_size, pc, params ); break;
case 432: arctanh_DD( task_size, pc, params ); break;
case 433: cos_FF( task_size, pc, params ); break;
case 434: cos_DD( task_size, pc, params ); break;
case 435: cosh_FF( task_size, pc, params ); break;
case 436: cosh_DD( task_size, pc, params ); break;
case 437: sin_FF( task_size, pc, params ); break;
case 438: sin_DD( task_size, pc, params ); break;
case 439: sinh_FF( task_size, pc, params ); break;
case 440: sinh_DD( task_size, pc, params ); break;
case 441: tan_FF( task_size, pc, params ); break;
case 442: tan_DD( task_size, pc, params ); break;
case 443: tanh_FF( task_size, pc, params ); break;
case 444: tanh_DD( task_size, pc, params ); break;
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


