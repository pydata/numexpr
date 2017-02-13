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

        // Sample debug output: 
        // printf( "Arg1(%d@%p) + ARG2(%d@%p) => DEST(%d@%p)\n", arg1, x1, arg2, x2, store_in, dest );
        
        // TODO: BLOCK_SIZE1 is the number of operations, not the array block size,
        // so the memory block scales with itemsize...
        //printf( "Exec op: %d\n", op );
        switch (params->program[pc].op) {
        case 0: 
            break;
        case 1: cast_b1( block_size, pc, params ); break;
case 2: cast_h1( block_size, pc, params ); break;
case 3: cast_i1( block_size, pc, params ); break;
case 4: cast_l1( block_size, pc, params ); break;
case 5: cast_B1( block_size, pc, params ); break;
case 6: cast_H1( block_size, pc, params ); break;
case 7: cast_I1( block_size, pc, params ); break;
case 8: cast_L1( block_size, pc, params ); break;
case 9: cast_f1( block_size, pc, params ); break;
case 10: cast_d1( block_size, pc, params ); break;
case 11: cast_F1( block_size, pc, params ); break;
case 12: cast_D1( block_size, pc, params ); break;
case 13: cast_hb( block_size, pc, params ); break;
case 14: cast_ib( block_size, pc, params ); break;
case 15: cast_lb( block_size, pc, params ); break;
case 16: cast_fb( block_size, pc, params ); break;
case 17: cast_db( block_size, pc, params ); break;
case 18: cast_Fb( block_size, pc, params ); break;
case 19: cast_Db( block_size, pc, params ); break;
case 20: cast_ih( block_size, pc, params ); break;
case 21: cast_lh( block_size, pc, params ); break;
case 22: cast_fh( block_size, pc, params ); break;
case 23: cast_dh( block_size, pc, params ); break;
case 24: cast_Fh( block_size, pc, params ); break;
case 25: cast_Dh( block_size, pc, params ); break;
case 26: cast_li( block_size, pc, params ); break;
case 27: cast_di( block_size, pc, params ); break;
case 28: cast_Di( block_size, pc, params ); break;
case 29: cast_dl( block_size, pc, params ); break;
case 30: cast_Dl( block_size, pc, params ); break;
case 31: cast_hB( block_size, pc, params ); break;
case 32: cast_iB( block_size, pc, params ); break;
case 33: cast_lB( block_size, pc, params ); break;
case 34: cast_HB( block_size, pc, params ); break;
case 35: cast_IB( block_size, pc, params ); break;
case 36: cast_LB( block_size, pc, params ); break;
case 37: cast_fB( block_size, pc, params ); break;
case 38: cast_dB( block_size, pc, params ); break;
case 39: cast_FB( block_size, pc, params ); break;
case 40: cast_DB( block_size, pc, params ); break;
case 41: cast_iH( block_size, pc, params ); break;
case 42: cast_lH( block_size, pc, params ); break;
case 43: cast_IH( block_size, pc, params ); break;
case 44: cast_LH( block_size, pc, params ); break;
case 45: cast_fH( block_size, pc, params ); break;
case 46: cast_dH( block_size, pc, params ); break;
case 47: cast_FH( block_size, pc, params ); break;
case 48: cast_DH( block_size, pc, params ); break;
case 49: cast_lI( block_size, pc, params ); break;
case 50: cast_LI( block_size, pc, params ); break;
case 51: cast_dI( block_size, pc, params ); break;
case 52: cast_DI( block_size, pc, params ); break;
case 53: cast_dL( block_size, pc, params ); break;
case 54: cast_DL( block_size, pc, params ); break;
case 55: cast_df( block_size, pc, params ); break;
case 56: cast_Ff( block_size, pc, params ); break;
case 57: cast_Df( block_size, pc, params ); break;
case 58: cast_Dd( block_size, pc, params ); break;
case 59: cast_DF( block_size, pc, params ); break;
case 60: copy_11( block_size, pc, params ); break;
case 61: copy_bb( block_size, pc, params ); break;
case 62: copy_hh( block_size, pc, params ); break;
case 63: copy_ii( block_size, pc, params ); break;
case 64: copy_ll( block_size, pc, params ); break;
case 65: copy_BB( block_size, pc, params ); break;
case 66: copy_HH( block_size, pc, params ); break;
case 67: copy_II( block_size, pc, params ); break;
case 68: copy_LL( block_size, pc, params ); break;
case 69: copy_ff( block_size, pc, params ); break;
case 70: copy_dd( block_size, pc, params ); break;
case 71: copy_FF( block_size, pc, params ); break;
case 72: copy_DD( block_size, pc, params ); break;
case 73: add_111( block_size, pc, params ); break;
case 74: add_bbb( block_size, pc, params ); break;
case 75: add_hhh( block_size, pc, params ); break;
case 76: add_iii( block_size, pc, params ); break;
case 77: add_lll( block_size, pc, params ); break;
case 78: add_BBB( block_size, pc, params ); break;
case 79: add_HHH( block_size, pc, params ); break;
case 80: add_III( block_size, pc, params ); break;
case 81: add_LLL( block_size, pc, params ); break;
case 82: add_fff( block_size, pc, params ); break;
case 83: add_ddd( block_size, pc, params ); break;
case 84: sub_bbb( block_size, pc, params ); break;
case 85: sub_hhh( block_size, pc, params ); break;
case 86: sub_iii( block_size, pc, params ); break;
case 87: sub_lll( block_size, pc, params ); break;
case 88: sub_BBB( block_size, pc, params ); break;
case 89: sub_HHH( block_size, pc, params ); break;
case 90: sub_III( block_size, pc, params ); break;
case 91: sub_LLL( block_size, pc, params ); break;
case 92: sub_fff( block_size, pc, params ); break;
case 93: sub_ddd( block_size, pc, params ); break;
case 94: mult_111( block_size, pc, params ); break;
case 95: mult_bbb( block_size, pc, params ); break;
case 96: mult_hhh( block_size, pc, params ); break;
case 97: mult_iii( block_size, pc, params ); break;
case 98: mult_lll( block_size, pc, params ); break;
case 99: mult_BBB( block_size, pc, params ); break;
case 100: mult_HHH( block_size, pc, params ); break;
case 101: mult_III( block_size, pc, params ); break;
case 102: mult_LLL( block_size, pc, params ); break;
case 103: mult_fff( block_size, pc, params ); break;
case 104: mult_ddd( block_size, pc, params ); break;
case 105: div_d11( block_size, pc, params ); break;
case 106: div_dbb( block_size, pc, params ); break;
case 107: div_dhh( block_size, pc, params ); break;
case 108: div_dii( block_size, pc, params ); break;
case 109: div_dll( block_size, pc, params ); break;
case 110: div_dBB( block_size, pc, params ); break;
case 111: div_dHH( block_size, pc, params ); break;
case 112: div_dII( block_size, pc, params ); break;
case 113: div_dLL( block_size, pc, params ); break;
case 114: div_fff( block_size, pc, params ); break;
case 115: div_ddd( block_size, pc, params ); break;
case 116: pow_fff( block_size, pc, params ); break;
case 117: pow_ddd( block_size, pc, params ); break;
case 118: mod_fff( block_size, pc, params ); break;
case 119: mod_ddd( block_size, pc, params ); break;
case 120: where_1111( block_size, pc, params ); break;
case 121: where_b1bb( block_size, pc, params ); break;
case 122: where_h1hh( block_size, pc, params ); break;
case 123: where_i1ii( block_size, pc, params ); break;
case 124: where_l1ll( block_size, pc, params ); break;
case 125: where_B1BB( block_size, pc, params ); break;
case 126: where_H1HH( block_size, pc, params ); break;
case 127: where_I1II( block_size, pc, params ); break;
case 128: where_L1LL( block_size, pc, params ); break;
case 129: where_f1ff( block_size, pc, params ); break;
case 130: where_d1dd( block_size, pc, params ); break;
case 131: where_F1FF( block_size, pc, params ); break;
case 132: where_D1DD( block_size, pc, params ); break;
case 133: ones_like_11( block_size, pc, params ); break;
case 134: ones_like_bb( block_size, pc, params ); break;
case 135: ones_like_hh( block_size, pc, params ); break;
case 136: ones_like_ii( block_size, pc, params ); break;
case 137: ones_like_ll( block_size, pc, params ); break;
case 138: ones_like_BB( block_size, pc, params ); break;
case 139: ones_like_HH( block_size, pc, params ); break;
case 140: ones_like_II( block_size, pc, params ); break;
case 141: ones_like_LL( block_size, pc, params ); break;
case 142: ones_like_ff( block_size, pc, params ); break;
case 143: ones_like_dd( block_size, pc, params ); break;
case 144: usub_bb( block_size, pc, params ); break;
case 145: usub_hh( block_size, pc, params ); break;
case 146: usub_ii( block_size, pc, params ); break;
case 147: usub_ll( block_size, pc, params ); break;
case 148: usub_ff( block_size, pc, params ); break;
case 149: usub_dd( block_size, pc, params ); break;
case 150: lshift_bbb( block_size, pc, params ); break;
case 151: lshift_hhh( block_size, pc, params ); break;
case 152: lshift_iii( block_size, pc, params ); break;
case 153: lshift_lll( block_size, pc, params ); break;
case 154: lshift_BBB( block_size, pc, params ); break;
case 155: lshift_HHH( block_size, pc, params ); break;
case 156: lshift_III( block_size, pc, params ); break;
case 157: lshift_LLL( block_size, pc, params ); break;
case 158: rshift_bbb( block_size, pc, params ); break;
case 159: rshift_hhh( block_size, pc, params ); break;
case 160: rshift_iii( block_size, pc, params ); break;
case 161: rshift_lll( block_size, pc, params ); break;
case 162: rshift_BBB( block_size, pc, params ); break;
case 163: rshift_HHH( block_size, pc, params ); break;
case 164: rshift_III( block_size, pc, params ); break;
case 165: rshift_LLL( block_size, pc, params ); break;
case 166: bitand_111( block_size, pc, params ); break;
case 167: bitand_bbb( block_size, pc, params ); break;
case 168: bitand_hhh( block_size, pc, params ); break;
case 169: bitand_iii( block_size, pc, params ); break;
case 170: bitand_lll( block_size, pc, params ); break;
case 171: bitand_BBB( block_size, pc, params ); break;
case 172: bitand_HHH( block_size, pc, params ); break;
case 173: bitand_III( block_size, pc, params ); break;
case 174: bitand_LLL( block_size, pc, params ); break;
case 175: bitor_111( block_size, pc, params ); break;
case 176: bitor_bbb( block_size, pc, params ); break;
case 177: bitor_hhh( block_size, pc, params ); break;
case 178: bitor_iii( block_size, pc, params ); break;
case 179: bitor_lll( block_size, pc, params ); break;
case 180: bitor_BBB( block_size, pc, params ); break;
case 181: bitor_HHH( block_size, pc, params ); break;
case 182: bitor_III( block_size, pc, params ); break;
case 183: bitor_LLL( block_size, pc, params ); break;
case 184: bitxor_111( block_size, pc, params ); break;
case 185: bitxor_bbb( block_size, pc, params ); break;
case 186: bitxor_hhh( block_size, pc, params ); break;
case 187: bitxor_iii( block_size, pc, params ); break;
case 188: bitxor_lll( block_size, pc, params ); break;
case 189: bitxor_BBB( block_size, pc, params ); break;
case 190: bitxor_HHH( block_size, pc, params ); break;
case 191: bitxor_III( block_size, pc, params ); break;
case 192: bitxor_LLL( block_size, pc, params ); break;
case 193: logical_and_111( block_size, pc, params ); break;
case 194: logical_or_111( block_size, pc, params ); break;
case 195: gt_111( block_size, pc, params ); break;
case 196: gt_1bb( block_size, pc, params ); break;
case 197: gt_1hh( block_size, pc, params ); break;
case 198: gt_1ii( block_size, pc, params ); break;
case 199: gt_1ll( block_size, pc, params ); break;
case 200: gt_1BB( block_size, pc, params ); break;
case 201: gt_1HH( block_size, pc, params ); break;
case 202: gt_1II( block_size, pc, params ); break;
case 203: gt_1LL( block_size, pc, params ); break;
case 204: gt_1ff( block_size, pc, params ); break;
case 205: gt_1dd( block_size, pc, params ); break;
case 206: gte_111( block_size, pc, params ); break;
case 207: gte_1bb( block_size, pc, params ); break;
case 208: gte_1hh( block_size, pc, params ); break;
case 209: gte_1ii( block_size, pc, params ); break;
case 210: gte_1ll( block_size, pc, params ); break;
case 211: gte_1BB( block_size, pc, params ); break;
case 212: gte_1HH( block_size, pc, params ); break;
case 213: gte_1II( block_size, pc, params ); break;
case 214: gte_1LL( block_size, pc, params ); break;
case 215: gte_1ff( block_size, pc, params ); break;
case 216: gte_1dd( block_size, pc, params ); break;
case 217: lt_111( block_size, pc, params ); break;
case 218: lt_1bb( block_size, pc, params ); break;
case 219: lt_1hh( block_size, pc, params ); break;
case 220: lt_1ii( block_size, pc, params ); break;
case 221: lt_1ll( block_size, pc, params ); break;
case 222: lt_1BB( block_size, pc, params ); break;
case 223: lt_1HH( block_size, pc, params ); break;
case 224: lt_1II( block_size, pc, params ); break;
case 225: lt_1LL( block_size, pc, params ); break;
case 226: lt_1ff( block_size, pc, params ); break;
case 227: lt_1dd( block_size, pc, params ); break;
case 228: lte_111( block_size, pc, params ); break;
case 229: lte_1bb( block_size, pc, params ); break;
case 230: lte_1hh( block_size, pc, params ); break;
case 231: lte_1ii( block_size, pc, params ); break;
case 232: lte_1ll( block_size, pc, params ); break;
case 233: lte_1BB( block_size, pc, params ); break;
case 234: lte_1HH( block_size, pc, params ); break;
case 235: lte_1II( block_size, pc, params ); break;
case 236: lte_1LL( block_size, pc, params ); break;
case 237: lte_1ff( block_size, pc, params ); break;
case 238: lte_1dd( block_size, pc, params ); break;
case 239: eq_111( block_size, pc, params ); break;
case 240: eq_1bb( block_size, pc, params ); break;
case 241: eq_1hh( block_size, pc, params ); break;
case 242: eq_1ii( block_size, pc, params ); break;
case 243: eq_1ll( block_size, pc, params ); break;
case 244: eq_1BB( block_size, pc, params ); break;
case 245: eq_1HH( block_size, pc, params ); break;
case 246: eq_1II( block_size, pc, params ); break;
case 247: eq_1LL( block_size, pc, params ); break;
case 248: eq_1ff( block_size, pc, params ); break;
case 249: eq_1dd( block_size, pc, params ); break;
case 250: noteq_111( block_size, pc, params ); break;
case 251: noteq_1bb( block_size, pc, params ); break;
case 252: noteq_1hh( block_size, pc, params ); break;
case 253: noteq_1ii( block_size, pc, params ); break;
case 254: noteq_1ll( block_size, pc, params ); break;
case 255: noteq_1BB( block_size, pc, params ); break;
case 256: noteq_1HH( block_size, pc, params ); break;
case 257: noteq_1II( block_size, pc, params ); break;
case 258: noteq_1LL( block_size, pc, params ); break;
case 259: noteq_1ff( block_size, pc, params ); break;
case 260: noteq_1dd( block_size, pc, params ); break;
case 261: abs_bb( block_size, pc, params ); break;
case 262: abs_hh( block_size, pc, params ); break;
case 263: abs_ii( block_size, pc, params ); break;
case 264: abs_ll( block_size, pc, params ); break;
case 265: abs_ff( block_size, pc, params ); break;
case 266: abs_dd( block_size, pc, params ); break;
case 267: arccos_ff( block_size, pc, params ); break;
case 268: arccos_dd( block_size, pc, params ); break;
case 269: arcsin_ff( block_size, pc, params ); break;
case 270: arcsin_dd( block_size, pc, params ); break;
case 271: arctan_ff( block_size, pc, params ); break;
case 272: arctan_dd( block_size, pc, params ); break;
case 273: arctan2_fff( block_size, pc, params ); break;
case 274: arctan2_ddd( block_size, pc, params ); break;
case 275: ceil_ff( block_size, pc, params ); break;
case 276: ceil_dd( block_size, pc, params ); break;
case 277: cos_ff( block_size, pc, params ); break;
case 278: cos_dd( block_size, pc, params ); break;
case 279: cosh_ff( block_size, pc, params ); break;
case 280: cosh_dd( block_size, pc, params ); break;
case 281: exp_ff( block_size, pc, params ); break;
case 282: exp_dd( block_size, pc, params ); break;
case 283: fabs_ff( block_size, pc, params ); break;
case 284: fabs_dd( block_size, pc, params ); break;
case 285: floor_ff( block_size, pc, params ); break;
case 286: floor_dd( block_size, pc, params ); break;
case 287: fmod_fff( block_size, pc, params ); break;
case 288: fmod_ddd( block_size, pc, params ); break;
case 289: log_ff( block_size, pc, params ); break;
case 290: log_dd( block_size, pc, params ); break;
case 291: log10_ff( block_size, pc, params ); break;
case 292: log10_dd( block_size, pc, params ); break;
case 293: sin_ff( block_size, pc, params ); break;
case 294: sin_dd( block_size, pc, params ); break;
case 295: sinh_ff( block_size, pc, params ); break;
case 296: sinh_dd( block_size, pc, params ); break;
case 297: sqrt_ff( block_size, pc, params ); break;
case 298: sqrt_dd( block_size, pc, params ); break;
case 299: tan_ff( block_size, pc, params ); break;
case 300: tan_dd( block_size, pc, params ); break;
case 301: tanh_ff( block_size, pc, params ); break;
case 302: tanh_dd( block_size, pc, params ); break;
case 303: fpclassify_if( block_size, pc, params ); break;
case 304: fpclassify_id( block_size, pc, params ); break;
case 305: isfinite_1f( block_size, pc, params ); break;
case 306: isfinite_1d( block_size, pc, params ); break;
case 307: isinf_1f( block_size, pc, params ); break;
case 308: isinf_1d( block_size, pc, params ); break;
case 309: isnan_1f( block_size, pc, params ); break;
case 310: isnan_1d( block_size, pc, params ); break;
case 311: isnormal_1f( block_size, pc, params ); break;
case 312: isnormal_1d( block_size, pc, params ); break;
case 313: signbit_1f( block_size, pc, params ); break;
case 314: signbit_1d( block_size, pc, params ); break;
case 315: arccosh_ff( block_size, pc, params ); break;
case 316: arccosh_dd( block_size, pc, params ); break;
case 317: arcsinh_ff( block_size, pc, params ); break;
case 318: arcsinh_dd( block_size, pc, params ); break;
case 319: arctanh_ff( block_size, pc, params ); break;
case 320: arctanh_dd( block_size, pc, params ); break;
case 321: cbrt_ff( block_size, pc, params ); break;
case 322: cbrt_dd( block_size, pc, params ); break;
case 323: copysign_fff( block_size, pc, params ); break;
case 324: copysign_ddd( block_size, pc, params ); break;
case 325: erf_ff( block_size, pc, params ); break;
case 326: erf_dd( block_size, pc, params ); break;
case 327: erfc_ff( block_size, pc, params ); break;
case 328: erfc_dd( block_size, pc, params ); break;
case 329: exp2_ff( block_size, pc, params ); break;
case 330: exp2_dd( block_size, pc, params ); break;
case 331: expm1_ff( block_size, pc, params ); break;
case 332: expm1_dd( block_size, pc, params ); break;
case 333: fdim_fff( block_size, pc, params ); break;
case 334: fdim_ddd( block_size, pc, params ); break;
case 335: fma_ffff( block_size, pc, params ); break;
case 336: fma_dddd( block_size, pc, params ); break;
case 337: fmax_fff( block_size, pc, params ); break;
case 338: fmax_ddd( block_size, pc, params ); break;
case 339: fmin_fff( block_size, pc, params ); break;
case 340: fmin_ddd( block_size, pc, params ); break;
case 341: hypot_fff( block_size, pc, params ); break;
case 342: hypot_ddd( block_size, pc, params ); break;
case 343: ilogb_if( block_size, pc, params ); break;
case 344: ilogb_id( block_size, pc, params ); break;
case 345: lgamma_ff( block_size, pc, params ); break;
case 346: lgamma_dd( block_size, pc, params ); break;
case 347: log1p_ff( block_size, pc, params ); break;
case 348: log1p_dd( block_size, pc, params ); break;
case 349: log2_ff( block_size, pc, params ); break;
case 350: log2_dd( block_size, pc, params ); break;
case 351: logb_ff( block_size, pc, params ); break;
case 352: logb_dd( block_size, pc, params ); break;
case 353: lrint_lf( block_size, pc, params ); break;
case 354: lrint_ld( block_size, pc, params ); break;
case 355: lround_lf( block_size, pc, params ); break;
case 356: lround_ld( block_size, pc, params ); break;
case 357: nearbyint_lf( block_size, pc, params ); break;
case 358: nearbyint_ld( block_size, pc, params ); break;
case 359: nextafter_fff( block_size, pc, params ); break;
case 360: nextafter_ddd( block_size, pc, params ); break;
case 361: nexttoward_fff( block_size, pc, params ); break;
case 362: nexttoward_ddd( block_size, pc, params ); break;
case 363: rint_ff( block_size, pc, params ); break;
case 364: rint_dd( block_size, pc, params ); break;
case 365: round_if( block_size, pc, params ); break;
case 366: round_id( block_size, pc, params ); break;
case 367: scalbln_ffl( block_size, pc, params ); break;
case 368: scalbln_ddl( block_size, pc, params ); break;
case 369: tgamma_ff( block_size, pc, params ); break;
case 370: tgamma_dd( block_size, pc, params ); break;
case 371: trunc_ff( block_size, pc, params ); break;
case 372: trunc_dd( block_size, pc, params ); break;
case 373: complex_Fff( block_size, pc, params ); break;
case 374: complex_Ddd( block_size, pc, params ); break;
case 375: real_fF( block_size, pc, params ); break;
case 376: real_dD( block_size, pc, params ); break;
case 377: imag_fF( block_size, pc, params ); break;
case 378: imag_dD( block_size, pc, params ); break;
case 379: abs_fF( block_size, pc, params ); break;
case 380: abs_dD( block_size, pc, params ); break;
case 381: abs2_fF( block_size, pc, params ); break;
case 382: abs2_dD( block_size, pc, params ); break;
case 383: add_FFF( block_size, pc, params ); break;
case 384: add_DDD( block_size, pc, params ); break;
case 385: sub_FFF( block_size, pc, params ); break;
case 386: sub_DDD( block_size, pc, params ); break;
case 387: mult_FFF( block_size, pc, params ); break;
case 388: mult_DDD( block_size, pc, params ); break;
case 389: div_FFF( block_size, pc, params ); break;
case 390: div_DDD( block_size, pc, params ); break;
case 391: usub_FF( block_size, pc, params ); break;
case 392: usub_DD( block_size, pc, params ); break;
case 393: neg_FF( block_size, pc, params ); break;
case 394: neg_DD( block_size, pc, params ); break;
case 395: conj_FF( block_size, pc, params ); break;
case 396: conj_DD( block_size, pc, params ); break;
case 397: conj_ff( block_size, pc, params ); break;
case 398: conj_dd( block_size, pc, params ); break;
case 399: sqrt_FF( block_size, pc, params ); break;
case 400: sqrt_DD( block_size, pc, params ); break;
case 401: log_FF( block_size, pc, params ); break;
case 402: log_DD( block_size, pc, params ); break;
case 403: log1p_FF( block_size, pc, params ); break;
case 404: log1p_DD( block_size, pc, params ); break;
case 405: log10_FF( block_size, pc, params ); break;
case 406: log10_DD( block_size, pc, params ); break;
case 407: exp_FF( block_size, pc, params ); break;
case 408: exp_DD( block_size, pc, params ); break;
case 409: expm1_FF( block_size, pc, params ); break;
case 410: expm1_DD( block_size, pc, params ); break;
case 411: pow_FFF( block_size, pc, params ); break;
case 412: pow_DDD( block_size, pc, params ); break;
case 413: arccos_FF( block_size, pc, params ); break;
case 414: arccos_DD( block_size, pc, params ); break;
case 415: arccosh_FF( block_size, pc, params ); break;
case 416: arccosh_DD( block_size, pc, params ); break;
case 417: arcsin_FF( block_size, pc, params ); break;
case 418: arcsin_DD( block_size, pc, params ); break;
case 419: arcsinh_FF( block_size, pc, params ); break;
case 420: arcsinh_DD( block_size, pc, params ); break;
case 421: arctan_FF( block_size, pc, params ); break;
case 422: arctan_DD( block_size, pc, params ); break;
case 423: arctanh_FF( block_size, pc, params ); break;
case 424: arctanh_DD( block_size, pc, params ); break;
case 425: cos_FF( block_size, pc, params ); break;
case 426: cos_DD( block_size, pc, params ); break;
case 427: cosh_FF( block_size, pc, params ); break;
case 428: cosh_DD( block_size, pc, params ); break;
case 429: sin_FF( block_size, pc, params ); break;
case 430: sin_DD( block_size, pc, params ); break;
case 431: sinh_FF( block_size, pc, params ); break;
case 432: sinh_DD( block_size, pc, params ); break;
case 433: tan_FF( block_size, pc, params ); break;
case 434: tan_DD( block_size, pc, params ); break;
case 435: tanh_FF( block_size, pc, params ); break;
case 436: tanh_DD( block_size, pc, params ); break;
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


