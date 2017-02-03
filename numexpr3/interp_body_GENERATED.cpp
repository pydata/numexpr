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
#else // SINGLE_ITEM_CONST_LOOP
    npy_intp J;
    // use the iterator's inner loop data
    // RAM: TODO, copy iterDataPtr into the params->registers[:].mem
    //memcpy(mem, iterDataPtr, (1+params->n_ndarray)*sizeof(npy_intp));
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


#endif // SINGLE_ITEM_CONST_LOOP

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
        case 1: cast_11( BLOCK_SIZE, pc, params ); break;
case 2: cast_b1( BLOCK_SIZE, pc, params ); break;
case 3: cast_h1( BLOCK_SIZE, pc, params ); break;
case 4: cast_i1( BLOCK_SIZE, pc, params ); break;
case 5: cast_l1( BLOCK_SIZE, pc, params ); break;
case 6: cast_B1( BLOCK_SIZE, pc, params ); break;
case 7: cast_H1( BLOCK_SIZE, pc, params ); break;
case 8: cast_I1( BLOCK_SIZE, pc, params ); break;
case 9: cast_L1( BLOCK_SIZE, pc, params ); break;
case 10: cast_f1( BLOCK_SIZE, pc, params ); break;
case 11: cast_d1( BLOCK_SIZE, pc, params ); break;
case 12: cast_bb( BLOCK_SIZE, pc, params ); break;
case 13: cast_hb( BLOCK_SIZE, pc, params ); break;
case 14: cast_ib( BLOCK_SIZE, pc, params ); break;
case 15: cast_lb( BLOCK_SIZE, pc, params ); break;
case 16: cast_fb( BLOCK_SIZE, pc, params ); break;
case 17: cast_db( BLOCK_SIZE, pc, params ); break;
case 18: cast_hh( BLOCK_SIZE, pc, params ); break;
case 19: cast_ih( BLOCK_SIZE, pc, params ); break;
case 20: cast_lh( BLOCK_SIZE, pc, params ); break;
case 21: cast_fh( BLOCK_SIZE, pc, params ); break;
case 22: cast_dh( BLOCK_SIZE, pc, params ); break;
case 23: cast_ii( BLOCK_SIZE, pc, params ); break;
case 24: cast_li( BLOCK_SIZE, pc, params ); break;
case 25: cast_di( BLOCK_SIZE, pc, params ); break;
case 26: cast_ll( BLOCK_SIZE, pc, params ); break;
case 27: cast_dl( BLOCK_SIZE, pc, params ); break;
case 28: cast_hB( BLOCK_SIZE, pc, params ); break;
case 29: cast_iB( BLOCK_SIZE, pc, params ); break;
case 30: cast_lB( BLOCK_SIZE, pc, params ); break;
case 31: cast_BB( BLOCK_SIZE, pc, params ); break;
case 32: cast_HB( BLOCK_SIZE, pc, params ); break;
case 33: cast_IB( BLOCK_SIZE, pc, params ); break;
case 34: cast_LB( BLOCK_SIZE, pc, params ); break;
case 35: cast_fB( BLOCK_SIZE, pc, params ); break;
case 36: cast_dB( BLOCK_SIZE, pc, params ); break;
case 37: cast_iH( BLOCK_SIZE, pc, params ); break;
case 38: cast_lH( BLOCK_SIZE, pc, params ); break;
case 39: cast_HH( BLOCK_SIZE, pc, params ); break;
case 40: cast_IH( BLOCK_SIZE, pc, params ); break;
case 41: cast_LH( BLOCK_SIZE, pc, params ); break;
case 42: cast_fH( BLOCK_SIZE, pc, params ); break;
case 43: cast_dH( BLOCK_SIZE, pc, params ); break;
case 44: cast_lI( BLOCK_SIZE, pc, params ); break;
case 45: cast_II( BLOCK_SIZE, pc, params ); break;
case 46: cast_LI( BLOCK_SIZE, pc, params ); break;
case 47: cast_dI( BLOCK_SIZE, pc, params ); break;
case 48: cast_LL( BLOCK_SIZE, pc, params ); break;
case 49: cast_dL( BLOCK_SIZE, pc, params ); break;
case 50: cast_ff( BLOCK_SIZE, pc, params ); break;
case 51: cast_df( BLOCK_SIZE, pc, params ); break;
case 52: cast_dd( BLOCK_SIZE, pc, params ); break;
case 53: copy_11( BLOCK_SIZE, pc, params ); break;
case 54: copy_bb( BLOCK_SIZE, pc, params ); break;
case 55: copy_hh( BLOCK_SIZE, pc, params ); break;
case 56: copy_ii( BLOCK_SIZE, pc, params ); break;
case 57: copy_ll( BLOCK_SIZE, pc, params ); break;
case 58: copy_BB( BLOCK_SIZE, pc, params ); break;
case 59: copy_HH( BLOCK_SIZE, pc, params ); break;
case 60: copy_II( BLOCK_SIZE, pc, params ); break;
case 61: copy_LL( BLOCK_SIZE, pc, params ); break;
case 62: copy_ff( BLOCK_SIZE, pc, params ); break;
case 63: copy_dd( BLOCK_SIZE, pc, params ); break;
case 64: copy_FF( BLOCK_SIZE, pc, params ); break;
case 65: copy_DD( BLOCK_SIZE, pc, params ); break;
case 66: add_111( BLOCK_SIZE, pc, params ); break;
case 67: add_bbb( BLOCK_SIZE, pc, params ); break;
case 68: add_hhh( BLOCK_SIZE, pc, params ); break;
case 69: add_iii( BLOCK_SIZE, pc, params ); break;
case 70: add_lll( BLOCK_SIZE, pc, params ); break;
case 71: add_BBB( BLOCK_SIZE, pc, params ); break;
case 72: add_HHH( BLOCK_SIZE, pc, params ); break;
case 73: add_III( BLOCK_SIZE, pc, params ); break;
case 74: add_LLL( BLOCK_SIZE, pc, params ); break;
case 75: add_fff( BLOCK_SIZE, pc, params ); break;
case 76: add_ddd( BLOCK_SIZE, pc, params ); break;
case 77: sub_111( BLOCK_SIZE, pc, params ); break;
case 78: sub_bbb( BLOCK_SIZE, pc, params ); break;
case 79: sub_hhh( BLOCK_SIZE, pc, params ); break;
case 80: sub_iii( BLOCK_SIZE, pc, params ); break;
case 81: sub_lll( BLOCK_SIZE, pc, params ); break;
case 82: sub_BBB( BLOCK_SIZE, pc, params ); break;
case 83: sub_HHH( BLOCK_SIZE, pc, params ); break;
case 84: sub_III( BLOCK_SIZE, pc, params ); break;
case 85: sub_LLL( BLOCK_SIZE, pc, params ); break;
case 86: sub_fff( BLOCK_SIZE, pc, params ); break;
case 87: sub_ddd( BLOCK_SIZE, pc, params ); break;
case 88: mult_111( BLOCK_SIZE, pc, params ); break;
case 89: mult_bbb( BLOCK_SIZE, pc, params ); break;
case 90: mult_hhh( BLOCK_SIZE, pc, params ); break;
case 91: mult_iii( BLOCK_SIZE, pc, params ); break;
case 92: mult_lll( BLOCK_SIZE, pc, params ); break;
case 93: mult_BBB( BLOCK_SIZE, pc, params ); break;
case 94: mult_HHH( BLOCK_SIZE, pc, params ); break;
case 95: mult_III( BLOCK_SIZE, pc, params ); break;
case 96: mult_LLL( BLOCK_SIZE, pc, params ); break;
case 97: mult_fff( BLOCK_SIZE, pc, params ); break;
case 98: mult_ddd( BLOCK_SIZE, pc, params ); break;
case 99: div_111( BLOCK_SIZE, pc, params ); break;
case 100: div_bbb( BLOCK_SIZE, pc, params ); break;
case 101: div_hhh( BLOCK_SIZE, pc, params ); break;
case 102: div_iii( BLOCK_SIZE, pc, params ); break;
case 103: div_lll( BLOCK_SIZE, pc, params ); break;
case 104: div_BBB( BLOCK_SIZE, pc, params ); break;
case 105: div_HHH( BLOCK_SIZE, pc, params ); break;
case 106: div_III( BLOCK_SIZE, pc, params ); break;
case 107: div_LLL( BLOCK_SIZE, pc, params ); break;
case 108: div_fff( BLOCK_SIZE, pc, params ); break;
case 109: div_ddd( BLOCK_SIZE, pc, params ); break;
case 110: pow_fff( BLOCK_SIZE, pc, params ); break;
case 111: pow_ddd( BLOCK_SIZE, pc, params ); break;
case 112: mod_111( BLOCK_SIZE, pc, params ); break;
case 113: mod_bbb( BLOCK_SIZE, pc, params ); break;
case 114: mod_hhh( BLOCK_SIZE, pc, params ); break;
case 115: mod_iii( BLOCK_SIZE, pc, params ); break;
case 116: mod_lll( BLOCK_SIZE, pc, params ); break;
case 117: mod_BBB( BLOCK_SIZE, pc, params ); break;
case 118: mod_HHH( BLOCK_SIZE, pc, params ); break;
case 119: mod_III( BLOCK_SIZE, pc, params ); break;
case 120: mod_LLL( BLOCK_SIZE, pc, params ); break;
case 121: mod_fff( BLOCK_SIZE, pc, params ); break;
case 122: mod_ddd( BLOCK_SIZE, pc, params ); break;
case 123: where_1111( BLOCK_SIZE, pc, params ); break;
case 124: where_b1bb( BLOCK_SIZE, pc, params ); break;
case 125: where_h1hh( BLOCK_SIZE, pc, params ); break;
case 126: where_i1ii( BLOCK_SIZE, pc, params ); break;
case 127: where_l1ll( BLOCK_SIZE, pc, params ); break;
case 128: where_B1BB( BLOCK_SIZE, pc, params ); break;
case 129: where_H1HH( BLOCK_SIZE, pc, params ); break;
case 130: where_I1II( BLOCK_SIZE, pc, params ); break;
case 131: where_L1LL( BLOCK_SIZE, pc, params ); break;
case 132: where_f1ff( BLOCK_SIZE, pc, params ); break;
case 133: where_d1dd( BLOCK_SIZE, pc, params ); break;
case 134: ones_like_1( BLOCK_SIZE, pc, params ); break;
case 135: ones_like_b( BLOCK_SIZE, pc, params ); break;
case 136: ones_like_h( BLOCK_SIZE, pc, params ); break;
case 137: ones_like_i( BLOCK_SIZE, pc, params ); break;
case 138: ones_like_l( BLOCK_SIZE, pc, params ); break;
case 139: ones_like_B( BLOCK_SIZE, pc, params ); break;
case 140: ones_like_H( BLOCK_SIZE, pc, params ); break;
case 141: ones_like_I( BLOCK_SIZE, pc, params ); break;
case 142: ones_like_L( BLOCK_SIZE, pc, params ); break;
case 143: ones_like_f( BLOCK_SIZE, pc, params ); break;
case 144: ones_like_d( BLOCK_SIZE, pc, params ); break;
case 145: neg_bb( BLOCK_SIZE, pc, params ); break;
case 146: neg_hh( BLOCK_SIZE, pc, params ); break;
case 147: neg_ii( BLOCK_SIZE, pc, params ); break;
case 148: neg_ll( BLOCK_SIZE, pc, params ); break;
case 149: neg_ff( BLOCK_SIZE, pc, params ); break;
case 150: neg_dd( BLOCK_SIZE, pc, params ); break;
case 151: lshift_bbb( BLOCK_SIZE, pc, params ); break;
case 152: lshift_hhh( BLOCK_SIZE, pc, params ); break;
case 153: lshift_iii( BLOCK_SIZE, pc, params ); break;
case 154: lshift_lll( BLOCK_SIZE, pc, params ); break;
case 155: lshift_BBB( BLOCK_SIZE, pc, params ); break;
case 156: lshift_HHH( BLOCK_SIZE, pc, params ); break;
case 157: lshift_III( BLOCK_SIZE, pc, params ); break;
case 158: lshift_LLL( BLOCK_SIZE, pc, params ); break;
case 159: rshift_bbb( BLOCK_SIZE, pc, params ); break;
case 160: rshift_hhh( BLOCK_SIZE, pc, params ); break;
case 161: rshift_iii( BLOCK_SIZE, pc, params ); break;
case 162: rshift_lll( BLOCK_SIZE, pc, params ); break;
case 163: rshift_BBB( BLOCK_SIZE, pc, params ); break;
case 164: rshift_HHH( BLOCK_SIZE, pc, params ); break;
case 165: rshift_III( BLOCK_SIZE, pc, params ); break;
case 166: rshift_LLL( BLOCK_SIZE, pc, params ); break;
case 167: bitand_111( BLOCK_SIZE, pc, params ); break;
case 168: bitand_bbb( BLOCK_SIZE, pc, params ); break;
case 169: bitand_hhh( BLOCK_SIZE, pc, params ); break;
case 170: bitand_iii( BLOCK_SIZE, pc, params ); break;
case 171: bitand_lll( BLOCK_SIZE, pc, params ); break;
case 172: bitand_BBB( BLOCK_SIZE, pc, params ); break;
case 173: bitand_HHH( BLOCK_SIZE, pc, params ); break;
case 174: bitand_III( BLOCK_SIZE, pc, params ); break;
case 175: bitand_LLL( BLOCK_SIZE, pc, params ); break;
case 176: bitor_111( BLOCK_SIZE, pc, params ); break;
case 177: bitor_bbb( BLOCK_SIZE, pc, params ); break;
case 178: bitor_hhh( BLOCK_SIZE, pc, params ); break;
case 179: bitor_iii( BLOCK_SIZE, pc, params ); break;
case 180: bitor_lll( BLOCK_SIZE, pc, params ); break;
case 181: bitor_BBB( BLOCK_SIZE, pc, params ); break;
case 182: bitor_HHH( BLOCK_SIZE, pc, params ); break;
case 183: bitor_III( BLOCK_SIZE, pc, params ); break;
case 184: bitor_LLL( BLOCK_SIZE, pc, params ); break;
case 185: bitxor_111( BLOCK_SIZE, pc, params ); break;
case 186: bitxor_bbb( BLOCK_SIZE, pc, params ); break;
case 187: bitxor_hhh( BLOCK_SIZE, pc, params ); break;
case 188: bitxor_iii( BLOCK_SIZE, pc, params ); break;
case 189: bitxor_lll( BLOCK_SIZE, pc, params ); break;
case 190: bitxor_BBB( BLOCK_SIZE, pc, params ); break;
case 191: bitxor_HHH( BLOCK_SIZE, pc, params ); break;
case 192: bitxor_III( BLOCK_SIZE, pc, params ); break;
case 193: bitxor_LLL( BLOCK_SIZE, pc, params ); break;
case 194: and_111( BLOCK_SIZE, pc, params ); break;
case 195: or_111( BLOCK_SIZE, pc, params ); break;
case 196: gt_111( BLOCK_SIZE, pc, params ); break;
case 197: gt_bbb( BLOCK_SIZE, pc, params ); break;
case 198: gt_hhh( BLOCK_SIZE, pc, params ); break;
case 199: gt_iii( BLOCK_SIZE, pc, params ); break;
case 200: gt_lll( BLOCK_SIZE, pc, params ); break;
case 201: gt_BBB( BLOCK_SIZE, pc, params ); break;
case 202: gt_HHH( BLOCK_SIZE, pc, params ); break;
case 203: gt_III( BLOCK_SIZE, pc, params ); break;
case 204: gt_LLL( BLOCK_SIZE, pc, params ); break;
case 205: gt_fff( BLOCK_SIZE, pc, params ); break;
case 206: gt_ddd( BLOCK_SIZE, pc, params ); break;
case 207: gte_111( BLOCK_SIZE, pc, params ); break;
case 208: gte_bbb( BLOCK_SIZE, pc, params ); break;
case 209: gte_hhh( BLOCK_SIZE, pc, params ); break;
case 210: gte_iii( BLOCK_SIZE, pc, params ); break;
case 211: gte_lll( BLOCK_SIZE, pc, params ); break;
case 212: gte_BBB( BLOCK_SIZE, pc, params ); break;
case 213: gte_HHH( BLOCK_SIZE, pc, params ); break;
case 214: gte_III( BLOCK_SIZE, pc, params ); break;
case 215: gte_LLL( BLOCK_SIZE, pc, params ); break;
case 216: gte_fff( BLOCK_SIZE, pc, params ); break;
case 217: gte_ddd( BLOCK_SIZE, pc, params ); break;
case 218: lt_111( BLOCK_SIZE, pc, params ); break;
case 219: lt_bbb( BLOCK_SIZE, pc, params ); break;
case 220: lt_hhh( BLOCK_SIZE, pc, params ); break;
case 221: lt_iii( BLOCK_SIZE, pc, params ); break;
case 222: lt_lll( BLOCK_SIZE, pc, params ); break;
case 223: lt_BBB( BLOCK_SIZE, pc, params ); break;
case 224: lt_HHH( BLOCK_SIZE, pc, params ); break;
case 225: lt_III( BLOCK_SIZE, pc, params ); break;
case 226: lt_LLL( BLOCK_SIZE, pc, params ); break;
case 227: lt_fff( BLOCK_SIZE, pc, params ); break;
case 228: lt_ddd( BLOCK_SIZE, pc, params ); break;
case 229: lte_111( BLOCK_SIZE, pc, params ); break;
case 230: lte_bbb( BLOCK_SIZE, pc, params ); break;
case 231: lte_hhh( BLOCK_SIZE, pc, params ); break;
case 232: lte_iii( BLOCK_SIZE, pc, params ); break;
case 233: lte_lll( BLOCK_SIZE, pc, params ); break;
case 234: lte_BBB( BLOCK_SIZE, pc, params ); break;
case 235: lte_HHH( BLOCK_SIZE, pc, params ); break;
case 236: lte_III( BLOCK_SIZE, pc, params ); break;
case 237: lte_LLL( BLOCK_SIZE, pc, params ); break;
case 238: lte_fff( BLOCK_SIZE, pc, params ); break;
case 239: lte_ddd( BLOCK_SIZE, pc, params ); break;
case 240: eq_111( BLOCK_SIZE, pc, params ); break;
case 241: eq_bbb( BLOCK_SIZE, pc, params ); break;
case 242: eq_hhh( BLOCK_SIZE, pc, params ); break;
case 243: eq_iii( BLOCK_SIZE, pc, params ); break;
case 244: eq_lll( BLOCK_SIZE, pc, params ); break;
case 245: eq_BBB( BLOCK_SIZE, pc, params ); break;
case 246: eq_HHH( BLOCK_SIZE, pc, params ); break;
case 247: eq_III( BLOCK_SIZE, pc, params ); break;
case 248: eq_LLL( BLOCK_SIZE, pc, params ); break;
case 249: eq_fff( BLOCK_SIZE, pc, params ); break;
case 250: eq_ddd( BLOCK_SIZE, pc, params ); break;
case 251: noteq_111( BLOCK_SIZE, pc, params ); break;
case 252: noteq_bbb( BLOCK_SIZE, pc, params ); break;
case 253: noteq_hhh( BLOCK_SIZE, pc, params ); break;
case 254: noteq_iii( BLOCK_SIZE, pc, params ); break;
case 255: noteq_lll( BLOCK_SIZE, pc, params ); break;
case 256: noteq_BBB( BLOCK_SIZE, pc, params ); break;
case 257: noteq_HHH( BLOCK_SIZE, pc, params ); break;
case 258: noteq_III( BLOCK_SIZE, pc, params ); break;
case 259: noteq_LLL( BLOCK_SIZE, pc, params ); break;
case 260: noteq_fff( BLOCK_SIZE, pc, params ); break;
case 261: noteq_ddd( BLOCK_SIZE, pc, params ); break;
case 262: abs_ff( BLOCK_SIZE, pc, params ); break;
case 263: abs_dd( BLOCK_SIZE, pc, params ); break;
case 264: arccos_ff( BLOCK_SIZE, pc, params ); break;
case 265: arccos_dd( BLOCK_SIZE, pc, params ); break;
case 266: arcsin_ff( BLOCK_SIZE, pc, params ); break;
case 267: arcsin_dd( BLOCK_SIZE, pc, params ); break;
case 268: arctan_ff( BLOCK_SIZE, pc, params ); break;
case 269: arctan_dd( BLOCK_SIZE, pc, params ); break;
case 270: arctan2_fff( BLOCK_SIZE, pc, params ); break;
case 271: arctan2_ddd( BLOCK_SIZE, pc, params ); break;
case 272: ceil_ff( BLOCK_SIZE, pc, params ); break;
case 273: ceil_dd( BLOCK_SIZE, pc, params ); break;
case 274: cos_ff( BLOCK_SIZE, pc, params ); break;
case 275: cos_dd( BLOCK_SIZE, pc, params ); break;
case 276: cosh_ff( BLOCK_SIZE, pc, params ); break;
case 277: cosh_dd( BLOCK_SIZE, pc, params ); break;
case 278: exp_ff( BLOCK_SIZE, pc, params ); break;
case 279: exp_dd( BLOCK_SIZE, pc, params ); break;
case 280: fabs_ff( BLOCK_SIZE, pc, params ); break;
case 281: fabs_dd( BLOCK_SIZE, pc, params ); break;
case 282: floor_ff( BLOCK_SIZE, pc, params ); break;
case 283: floor_dd( BLOCK_SIZE, pc, params ); break;
case 284: fmod_fff( BLOCK_SIZE, pc, params ); break;
case 285: fmod_ddd( BLOCK_SIZE, pc, params ); break;
case 286: log_ff( BLOCK_SIZE, pc, params ); break;
case 287: log_dd( BLOCK_SIZE, pc, params ); break;
case 288: log10_ff( BLOCK_SIZE, pc, params ); break;
case 289: log10_dd( BLOCK_SIZE, pc, params ); break;
case 290: power_fff( BLOCK_SIZE, pc, params ); break;
case 291: power_ddd( BLOCK_SIZE, pc, params ); break;
case 292: power_ffi( BLOCK_SIZE, pc, params ); break;
case 293: power_ddi( BLOCK_SIZE, pc, params ); break;
case 294: sin_ff( BLOCK_SIZE, pc, params ); break;
case 295: sin_dd( BLOCK_SIZE, pc, params ); break;
case 296: sinh_ff( BLOCK_SIZE, pc, params ); break;
case 297: sinh_dd( BLOCK_SIZE, pc, params ); break;
case 298: sqrt_ff( BLOCK_SIZE, pc, params ); break;
case 299: sqrt_dd( BLOCK_SIZE, pc, params ); break;
case 300: tan_ff( BLOCK_SIZE, pc, params ); break;
case 301: tan_dd( BLOCK_SIZE, pc, params ); break;
case 302: tanh_ff( BLOCK_SIZE, pc, params ); break;
case 303: tanh_dd( BLOCK_SIZE, pc, params ); break;
case 304: fpclassify_if( BLOCK_SIZE, pc, params ); break;
case 305: fpclassify_id( BLOCK_SIZE, pc, params ); break;
case 306: isfinite_1f( BLOCK_SIZE, pc, params ); break;
case 307: isfinite_1d( BLOCK_SIZE, pc, params ); break;
case 308: isinf_1f( BLOCK_SIZE, pc, params ); break;
case 309: isinf_1d( BLOCK_SIZE, pc, params ); break;
case 310: isnan_1f( BLOCK_SIZE, pc, params ); break;
case 311: isnan_1d( BLOCK_SIZE, pc, params ); break;
case 312: isnormal_1f( BLOCK_SIZE, pc, params ); break;
case 313: isnormal_1d( BLOCK_SIZE, pc, params ); break;
case 314: signbit_1f( BLOCK_SIZE, pc, params ); break;
case 315: signbit_1d( BLOCK_SIZE, pc, params ); break;
case 316: arccosh_ff( BLOCK_SIZE, pc, params ); break;
case 317: arccosh_dd( BLOCK_SIZE, pc, params ); break;
case 318: arcsinh_ff( BLOCK_SIZE, pc, params ); break;
case 319: arcsinh_dd( BLOCK_SIZE, pc, params ); break;
case 320: arctanh_ff( BLOCK_SIZE, pc, params ); break;
case 321: arctanh_dd( BLOCK_SIZE, pc, params ); break;
case 322: cbrt_ff( BLOCK_SIZE, pc, params ); break;
case 323: cbrt_dd( BLOCK_SIZE, pc, params ); break;
case 324: copysign_fff( BLOCK_SIZE, pc, params ); break;
case 325: copysign_ddd( BLOCK_SIZE, pc, params ); break;
case 326: erf_ff( BLOCK_SIZE, pc, params ); break;
case 327: erf_dd( BLOCK_SIZE, pc, params ); break;
case 328: erfc_ff( BLOCK_SIZE, pc, params ); break;
case 329: erfc_dd( BLOCK_SIZE, pc, params ); break;
case 330: exp2_ff( BLOCK_SIZE, pc, params ); break;
case 331: exp2_dd( BLOCK_SIZE, pc, params ); break;
case 332: expm1_ff( BLOCK_SIZE, pc, params ); break;
case 333: expm1_dd( BLOCK_SIZE, pc, params ); break;
case 334: fdim_fff( BLOCK_SIZE, pc, params ); break;
case 335: fdim_ddd( BLOCK_SIZE, pc, params ); break;
case 336: fma_ffff( BLOCK_SIZE, pc, params ); break;
case 337: fma_dddd( BLOCK_SIZE, pc, params ); break;
case 338: fmax_fff( BLOCK_SIZE, pc, params ); break;
case 339: fmax_ddd( BLOCK_SIZE, pc, params ); break;
case 340: fmin_fff( BLOCK_SIZE, pc, params ); break;
case 341: fmin_ddd( BLOCK_SIZE, pc, params ); break;
case 342: hypot_fff( BLOCK_SIZE, pc, params ); break;
case 343: hypot_ddd( BLOCK_SIZE, pc, params ); break;
case 344: ilogb_if( BLOCK_SIZE, pc, params ); break;
case 345: ilogb_id( BLOCK_SIZE, pc, params ); break;
case 346: lgamma_ff( BLOCK_SIZE, pc, params ); break;
case 347: lgamma_dd( BLOCK_SIZE, pc, params ); break;
case 348: log1p_ff( BLOCK_SIZE, pc, params ); break;
case 349: log1p_dd( BLOCK_SIZE, pc, params ); break;
case 350: log2_ff( BLOCK_SIZE, pc, params ); break;
case 351: log2_dd( BLOCK_SIZE, pc, params ); break;
case 352: logb_ff( BLOCK_SIZE, pc, params ); break;
case 353: logb_dd( BLOCK_SIZE, pc, params ); break;
case 354: lrint_lf( BLOCK_SIZE, pc, params ); break;
case 355: lrint_ld( BLOCK_SIZE, pc, params ); break;
case 356: lround_lf( BLOCK_SIZE, pc, params ); break;
case 357: lround_ld( BLOCK_SIZE, pc, params ); break;
case 358: nearbyint_lf( BLOCK_SIZE, pc, params ); break;
case 359: nearbyint_ld( BLOCK_SIZE, pc, params ); break;
case 360: nextafter_fff( BLOCK_SIZE, pc, params ); break;
case 361: nextafter_ddd( BLOCK_SIZE, pc, params ); break;
case 362: nexttoward_fff( BLOCK_SIZE, pc, params ); break;
case 363: nexttoward_ddd( BLOCK_SIZE, pc, params ); break;
case 364: remainder_fff( BLOCK_SIZE, pc, params ); break;
case 365: remainder_ddd( BLOCK_SIZE, pc, params ); break;
case 366: rint_ff( BLOCK_SIZE, pc, params ); break;
case 367: rint_dd( BLOCK_SIZE, pc, params ); break;
case 368: round_if( BLOCK_SIZE, pc, params ); break;
case 369: round_id( BLOCK_SIZE, pc, params ); break;
case 370: scalbln_ffl( BLOCK_SIZE, pc, params ); break;
case 371: scalbln_ddl( BLOCK_SIZE, pc, params ); break;
case 372: tgamma_ff( BLOCK_SIZE, pc, params ); break;
case 373: tgamma_dd( BLOCK_SIZE, pc, params ); break;
case 374: trunc_ff( BLOCK_SIZE, pc, params ); break;
case 375: trunc_dd( BLOCK_SIZE, pc, params ); break;
case 376: multest_ddd( BLOCK_SIZE, pc, params ); break;
case 377: abs_fF( BLOCK_SIZE, pc, params ); break;
case 378: abs_dD( BLOCK_SIZE, pc, params ); break;
case 379: add_FFF( BLOCK_SIZE, pc, params ); break;
case 380: add_DDD( BLOCK_SIZE, pc, params ); break;
case 381: sub_FFF( BLOCK_SIZE, pc, params ); break;
case 382: sub_DDD( BLOCK_SIZE, pc, params ); break;
case 383: mult_FFF( BLOCK_SIZE, pc, params ); break;
case 384: mult_DDD( BLOCK_SIZE, pc, params ); break;
case 385: div_FFF( BLOCK_SIZE, pc, params ); break;
case 386: div_DDD( BLOCK_SIZE, pc, params ); break;
case 387: neg_FF( BLOCK_SIZE, pc, params ); break;
case 388: neg_DD( BLOCK_SIZE, pc, params ); break;
case 389: conj_FF( BLOCK_SIZE, pc, params ); break;
case 390: conj_DD( BLOCK_SIZE, pc, params ); break;
case 391: conj_ff( BLOCK_SIZE, pc, params ); break;
case 392: conj_dd( BLOCK_SIZE, pc, params ); break;
case 393: sqrt_FF( BLOCK_SIZE, pc, params ); break;
case 394: sqrt_DD( BLOCK_SIZE, pc, params ); break;
case 395: log_FF( BLOCK_SIZE, pc, params ); break;
case 396: log_DD( BLOCK_SIZE, pc, params ); break;
case 397: log1p_FF( BLOCK_SIZE, pc, params ); break;
case 398: log1p_DD( BLOCK_SIZE, pc, params ); break;
case 399: log10_FF( BLOCK_SIZE, pc, params ); break;
case 400: log10_DD( BLOCK_SIZE, pc, params ); break;
case 401: exp_FF( BLOCK_SIZE, pc, params ); break;
case 402: exp_DD( BLOCK_SIZE, pc, params ); break;
case 403: expm1_FF( BLOCK_SIZE, pc, params ); break;
case 404: expm1_DD( BLOCK_SIZE, pc, params ); break;
case 405: pow_FFF( BLOCK_SIZE, pc, params ); break;
case 406: pow_DDD( BLOCK_SIZE, pc, params ); break;
case 407: arccos_FF( BLOCK_SIZE, pc, params ); break;
case 408: arccos_DD( BLOCK_SIZE, pc, params ); break;
case 409: arccosh_FF( BLOCK_SIZE, pc, params ); break;
case 410: arccosh_DD( BLOCK_SIZE, pc, params ); break;
case 411: arcsin_FF( BLOCK_SIZE, pc, params ); break;
case 412: arcsin_DD( BLOCK_SIZE, pc, params ); break;
case 413: arcsinh_FF( BLOCK_SIZE, pc, params ); break;
case 414: arcsinh_DD( BLOCK_SIZE, pc, params ); break;
case 415: arctan_FF( BLOCK_SIZE, pc, params ); break;
case 416: arctan_DD( BLOCK_SIZE, pc, params ); break;
case 417: arctanh_FF( BLOCK_SIZE, pc, params ); break;
case 418: arctanh_DD( BLOCK_SIZE, pc, params ); break;
case 419: cos_FF( BLOCK_SIZE, pc, params ); break;
case 420: cos_DD( BLOCK_SIZE, pc, params ); break;
case 421: cosh_FF( BLOCK_SIZE, pc, params ); break;
case 422: cosh_DD( BLOCK_SIZE, pc, params ); break;
case 423: sin_FF( BLOCK_SIZE, pc, params ); break;
case 424: sin_DD( BLOCK_SIZE, pc, params ); break;
case 425: sinh_FF( BLOCK_SIZE, pc, params ); break;
case 426: sinh_DD( BLOCK_SIZE, pc, params ); break;
case 427: tan_FF( BLOCK_SIZE, pc, params ); break;
case 428: tan_DD( BLOCK_SIZE, pc, params ); break;
case 429: tanh_FF( BLOCK_SIZE, pc, params ); break;
case 430: tanh_DD( BLOCK_SIZE, pc, params ); break;
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


