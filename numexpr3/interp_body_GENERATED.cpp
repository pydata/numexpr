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
        if( params->registers[J].kind & (KIND_ARRAY|KIND_RETURN) ) {
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
case 13: unsafe_cast_1b( task_size, pc, params ); break;
case 14: cast_hb( task_size, pc, params ); break;
case 15: cast_lb( task_size, pc, params ); break;
case 16: cast_qb( task_size, pc, params ); break;
case 17: unsafe_cast_Bb( task_size, pc, params ); break;
case 18: unsafe_cast_Hb( task_size, pc, params ); break;
case 19: unsafe_cast_Lb( task_size, pc, params ); break;
case 20: unsafe_cast_Qb( task_size, pc, params ); break;
case 21: cast_fb( task_size, pc, params ); break;
case 22: cast_db( task_size, pc, params ); break;
case 23: cast_Fb( task_size, pc, params ); break;
case 24: cast_Db( task_size, pc, params ); break;
case 25: unsafe_cast_1h( task_size, pc, params ); break;
case 26: unsafe_cast_bh( task_size, pc, params ); break;
case 27: cast_lh( task_size, pc, params ); break;
case 28: cast_qh( task_size, pc, params ); break;
case 29: unsafe_cast_Bh( task_size, pc, params ); break;
case 30: unsafe_cast_Hh( task_size, pc, params ); break;
case 31: unsafe_cast_Lh( task_size, pc, params ); break;
case 32: unsafe_cast_Qh( task_size, pc, params ); break;
case 33: cast_fh( task_size, pc, params ); break;
case 34: cast_dh( task_size, pc, params ); break;
case 35: cast_Fh( task_size, pc, params ); break;
case 36: cast_Dh( task_size, pc, params ); break;
case 37: unsafe_cast_1l( task_size, pc, params ); break;
case 38: unsafe_cast_bl( task_size, pc, params ); break;
case 39: unsafe_cast_hl( task_size, pc, params ); break;
case 40: cast_ql( task_size, pc, params ); break;
case 41: unsafe_cast_Bl( task_size, pc, params ); break;
case 42: unsafe_cast_Hl( task_size, pc, params ); break;
case 43: unsafe_cast_Ll( task_size, pc, params ); break;
case 44: unsafe_cast_Ql( task_size, pc, params ); break;
case 45: unsafe_cast_fl( task_size, pc, params ); break;
case 46: cast_dl( task_size, pc, params ); break;
case 47: cast_Dl( task_size, pc, params ); break;
case 48: unsafe_cast_1q( task_size, pc, params ); break;
case 49: unsafe_cast_bq( task_size, pc, params ); break;
case 50: unsafe_cast_hq( task_size, pc, params ); break;
case 51: unsafe_cast_lq( task_size, pc, params ); break;
case 52: unsafe_cast_Bq( task_size, pc, params ); break;
case 53: unsafe_cast_Hq( task_size, pc, params ); break;
case 54: unsafe_cast_Lq( task_size, pc, params ); break;
case 55: unsafe_cast_Qq( task_size, pc, params ); break;
case 56: unsafe_cast_fq( task_size, pc, params ); break;
case 57: cast_dq( task_size, pc, params ); break;
case 58: cast_Dq( task_size, pc, params ); break;
case 59: unsafe_cast_1B( task_size, pc, params ); break;
case 60: unsafe_cast_bB( task_size, pc, params ); break;
case 61: cast_hB( task_size, pc, params ); break;
case 62: cast_lB( task_size, pc, params ); break;
case 63: cast_qB( task_size, pc, params ); break;
case 64: cast_HB( task_size, pc, params ); break;
case 65: cast_LB( task_size, pc, params ); break;
case 66: cast_QB( task_size, pc, params ); break;
case 67: cast_fB( task_size, pc, params ); break;
case 68: cast_dB( task_size, pc, params ); break;
case 69: cast_FB( task_size, pc, params ); break;
case 70: cast_DB( task_size, pc, params ); break;
case 71: unsafe_cast_1H( task_size, pc, params ); break;
case 72: unsafe_cast_bH( task_size, pc, params ); break;
case 73: unsafe_cast_hH( task_size, pc, params ); break;
case 74: cast_lH( task_size, pc, params ); break;
case 75: cast_qH( task_size, pc, params ); break;
case 76: unsafe_cast_BH( task_size, pc, params ); break;
case 77: cast_LH( task_size, pc, params ); break;
case 78: cast_QH( task_size, pc, params ); break;
case 79: cast_fH( task_size, pc, params ); break;
case 80: cast_dH( task_size, pc, params ); break;
case 81: cast_FH( task_size, pc, params ); break;
case 82: cast_DH( task_size, pc, params ); break;
case 83: unsafe_cast_1L( task_size, pc, params ); break;
case 84: unsafe_cast_bL( task_size, pc, params ); break;
case 85: unsafe_cast_hL( task_size, pc, params ); break;
case 86: unsafe_cast_lL( task_size, pc, params ); break;
case 87: cast_qL( task_size, pc, params ); break;
case 88: unsafe_cast_BL( task_size, pc, params ); break;
case 89: unsafe_cast_HL( task_size, pc, params ); break;
case 90: cast_QL( task_size, pc, params ); break;
case 91: unsafe_cast_fL( task_size, pc, params ); break;
case 92: cast_dL( task_size, pc, params ); break;
case 93: cast_DL( task_size, pc, params ); break;
case 94: unsafe_cast_1Q( task_size, pc, params ); break;
case 95: unsafe_cast_bQ( task_size, pc, params ); break;
case 96: unsafe_cast_hQ( task_size, pc, params ); break;
case 97: unsafe_cast_lQ( task_size, pc, params ); break;
case 98: unsafe_cast_qQ( task_size, pc, params ); break;
case 99: unsafe_cast_BQ( task_size, pc, params ); break;
case 100: unsafe_cast_HQ( task_size, pc, params ); break;
case 101: unsafe_cast_LQ( task_size, pc, params ); break;
case 102: unsafe_cast_fQ( task_size, pc, params ); break;
case 103: cast_dQ( task_size, pc, params ); break;
case 104: cast_DQ( task_size, pc, params ); break;
case 105: unsafe_cast_1f( task_size, pc, params ); break;
case 106: unsafe_cast_bf( task_size, pc, params ); break;
case 107: unsafe_cast_hf( task_size, pc, params ); break;
case 108: unsafe_cast_lf( task_size, pc, params ); break;
case 109: unsafe_cast_qf( task_size, pc, params ); break;
case 110: unsafe_cast_Bf( task_size, pc, params ); break;
case 111: unsafe_cast_Hf( task_size, pc, params ); break;
case 112: unsafe_cast_Lf( task_size, pc, params ); break;
case 113: unsafe_cast_Qf( task_size, pc, params ); break;
case 114: cast_df( task_size, pc, params ); break;
case 115: cast_Ff( task_size, pc, params ); break;
case 116: cast_Df( task_size, pc, params ); break;
case 117: unsafe_cast_1d( task_size, pc, params ); break;
case 118: unsafe_cast_bd( task_size, pc, params ); break;
case 119: unsafe_cast_hd( task_size, pc, params ); break;
case 120: unsafe_cast_ld( task_size, pc, params ); break;
case 121: unsafe_cast_qd( task_size, pc, params ); break;
case 122: unsafe_cast_Bd( task_size, pc, params ); break;
case 123: unsafe_cast_Hd( task_size, pc, params ); break;
case 124: unsafe_cast_Ld( task_size, pc, params ); break;
case 125: unsafe_cast_Qd( task_size, pc, params ); break;
case 126: unsafe_cast_fd( task_size, pc, params ); break;
case 127: cast_Dd( task_size, pc, params ); break;
case 128: cast_DF( task_size, pc, params ); break;
case 129: copy_11( task_size, pc, params ); break;
case 130: copy_bb( task_size, pc, params ); break;
case 131: copy_hh( task_size, pc, params ); break;
case 132: copy_ll( task_size, pc, params ); break;
case 133: copy_qq( task_size, pc, params ); break;
case 134: copy_BB( task_size, pc, params ); break;
case 135: copy_HH( task_size, pc, params ); break;
case 136: copy_LL( task_size, pc, params ); break;
case 137: copy_QQ( task_size, pc, params ); break;
case 138: copy_ff( task_size, pc, params ); break;
case 139: copy_dd( task_size, pc, params ); break;
case 140: copy_FF( task_size, pc, params ); break;
case 141: copy_DD( task_size, pc, params ); break;
case 142: add_111( task_size, pc, params ); break;
case 143: add_bbb( task_size, pc, params ); break;
case 144: add_hhh( task_size, pc, params ); break;
case 145: add_lll( task_size, pc, params ); break;
case 146: add_qqq( task_size, pc, params ); break;
case 147: add_BBB( task_size, pc, params ); break;
case 148: add_HHH( task_size, pc, params ); break;
case 149: add_LLL( task_size, pc, params ); break;
case 150: add_QQQ( task_size, pc, params ); break;
case 151: add_fff( task_size, pc, params ); break;
case 152: add_ddd( task_size, pc, params ); break;
case 153: sub_bbb( task_size, pc, params ); break;
case 154: sub_hhh( task_size, pc, params ); break;
case 155: sub_lll( task_size, pc, params ); break;
case 156: sub_qqq( task_size, pc, params ); break;
case 157: sub_BBB( task_size, pc, params ); break;
case 158: sub_HHH( task_size, pc, params ); break;
case 159: sub_LLL( task_size, pc, params ); break;
case 160: sub_QQQ( task_size, pc, params ); break;
case 161: sub_fff( task_size, pc, params ); break;
case 162: sub_ddd( task_size, pc, params ); break;
case 163: mult_111( task_size, pc, params ); break;
case 164: mult_bbb( task_size, pc, params ); break;
case 165: mult_hhh( task_size, pc, params ); break;
case 166: mult_lll( task_size, pc, params ); break;
case 167: mult_qqq( task_size, pc, params ); break;
case 168: mult_BBB( task_size, pc, params ); break;
case 169: mult_HHH( task_size, pc, params ); break;
case 170: mult_LLL( task_size, pc, params ); break;
case 171: mult_QQQ( task_size, pc, params ); break;
case 172: mult_fff( task_size, pc, params ); break;
case 173: mult_ddd( task_size, pc, params ); break;
case 174: div_d11( task_size, pc, params ); break;
case 175: div_dbb( task_size, pc, params ); break;
case 176: div_dhh( task_size, pc, params ); break;
case 177: div_dll( task_size, pc, params ); break;
case 178: div_dqq( task_size, pc, params ); break;
case 179: div_dBB( task_size, pc, params ); break;
case 180: div_dHH( task_size, pc, params ); break;
case 181: div_dLL( task_size, pc, params ); break;
case 182: div_dQQ( task_size, pc, params ); break;
case 183: div_fff( task_size, pc, params ); break;
case 184: div_ddd( task_size, pc, params ); break;
case 185: floordiv_bbb( task_size, pc, params ); break;
case 186: floordiv_hhh( task_size, pc, params ); break;
case 187: floordiv_lll( task_size, pc, params ); break;
case 188: floordiv_qqq( task_size, pc, params ); break;
case 189: floordiv_BBB( task_size, pc, params ); break;
case 190: floordiv_HHH( task_size, pc, params ); break;
case 191: floordiv_LLL( task_size, pc, params ); break;
case 192: floordiv_QQQ( task_size, pc, params ); break;
case 193: pow_fff( task_size, pc, params ); break;
case 194: pow_ddd( task_size, pc, params ); break;
case 195: mod_fff( task_size, pc, params ); break;
case 196: mod_ddd( task_size, pc, params ); break;
case 197: where_1111( task_size, pc, params ); break;
case 198: where_b1bb( task_size, pc, params ); break;
case 199: where_h1hh( task_size, pc, params ); break;
case 200: where_l1ll( task_size, pc, params ); break;
case 201: where_q1qq( task_size, pc, params ); break;
case 202: where_B1BB( task_size, pc, params ); break;
case 203: where_H1HH( task_size, pc, params ); break;
case 204: where_L1LL( task_size, pc, params ); break;
case 205: where_Q1QQ( task_size, pc, params ); break;
case 206: where_f1ff( task_size, pc, params ); break;
case 207: where_d1dd( task_size, pc, params ); break;
case 208: where_F1FF( task_size, pc, params ); break;
case 209: where_D1DD( task_size, pc, params ); break;
case 210: ones_like_11( task_size, pc, params ); break;
case 211: ones_like_bb( task_size, pc, params ); break;
case 212: ones_like_hh( task_size, pc, params ); break;
case 213: ones_like_ll( task_size, pc, params ); break;
case 214: ones_like_qq( task_size, pc, params ); break;
case 215: ones_like_BB( task_size, pc, params ); break;
case 216: ones_like_HH( task_size, pc, params ); break;
case 217: ones_like_LL( task_size, pc, params ); break;
case 218: ones_like_QQ( task_size, pc, params ); break;
case 219: ones_like_ff( task_size, pc, params ); break;
case 220: ones_like_dd( task_size, pc, params ); break;
case 221: usub_bb( task_size, pc, params ); break;
case 222: usub_hh( task_size, pc, params ); break;
case 223: usub_ll( task_size, pc, params ); break;
case 224: usub_qq( task_size, pc, params ); break;
case 225: usub_ff( task_size, pc, params ); break;
case 226: usub_dd( task_size, pc, params ); break;
case 227: lshift_bbb( task_size, pc, params ); break;
case 228: lshift_hhh( task_size, pc, params ); break;
case 229: lshift_lll( task_size, pc, params ); break;
case 230: lshift_qqq( task_size, pc, params ); break;
case 231: lshift_BBB( task_size, pc, params ); break;
case 232: lshift_HHH( task_size, pc, params ); break;
case 233: lshift_LLL( task_size, pc, params ); break;
case 234: lshift_QQQ( task_size, pc, params ); break;
case 235: rshift_bbb( task_size, pc, params ); break;
case 236: rshift_hhh( task_size, pc, params ); break;
case 237: rshift_lll( task_size, pc, params ); break;
case 238: rshift_qqq( task_size, pc, params ); break;
case 239: rshift_BBB( task_size, pc, params ); break;
case 240: rshift_HHH( task_size, pc, params ); break;
case 241: rshift_LLL( task_size, pc, params ); break;
case 242: rshift_QQQ( task_size, pc, params ); break;
case 243: bitand_111( task_size, pc, params ); break;
case 244: bitand_bbb( task_size, pc, params ); break;
case 245: bitand_hhh( task_size, pc, params ); break;
case 246: bitand_lll( task_size, pc, params ); break;
case 247: bitand_qqq( task_size, pc, params ); break;
case 248: bitand_BBB( task_size, pc, params ); break;
case 249: bitand_HHH( task_size, pc, params ); break;
case 250: bitand_LLL( task_size, pc, params ); break;
case 251: bitand_QQQ( task_size, pc, params ); break;
case 252: bitor_111( task_size, pc, params ); break;
case 253: bitor_bbb( task_size, pc, params ); break;
case 254: bitor_hhh( task_size, pc, params ); break;
case 255: bitor_lll( task_size, pc, params ); break;
case 256: bitor_qqq( task_size, pc, params ); break;
case 257: bitor_BBB( task_size, pc, params ); break;
case 258: bitor_HHH( task_size, pc, params ); break;
case 259: bitor_LLL( task_size, pc, params ); break;
case 260: bitor_QQQ( task_size, pc, params ); break;
case 261: bitxor_111( task_size, pc, params ); break;
case 262: bitxor_bbb( task_size, pc, params ); break;
case 263: bitxor_hhh( task_size, pc, params ); break;
case 264: bitxor_lll( task_size, pc, params ); break;
case 265: bitxor_qqq( task_size, pc, params ); break;
case 266: bitxor_BBB( task_size, pc, params ); break;
case 267: bitxor_HHH( task_size, pc, params ); break;
case 268: bitxor_LLL( task_size, pc, params ); break;
case 269: bitxor_QQQ( task_size, pc, params ); break;
case 270: logical_and_111( task_size, pc, params ); break;
case 271: logical_or_111( task_size, pc, params ); break;
case 272: gt_111( task_size, pc, params ); break;
case 273: gt_1bb( task_size, pc, params ); break;
case 274: gt_1hh( task_size, pc, params ); break;
case 275: gt_1ll( task_size, pc, params ); break;
case 276: gt_1qq( task_size, pc, params ); break;
case 277: gt_1BB( task_size, pc, params ); break;
case 278: gt_1HH( task_size, pc, params ); break;
case 279: gt_1LL( task_size, pc, params ); break;
case 280: gt_1QQ( task_size, pc, params ); break;
case 281: gt_1ff( task_size, pc, params ); break;
case 282: gt_1dd( task_size, pc, params ); break;
case 283: gte_111( task_size, pc, params ); break;
case 284: gte_1bb( task_size, pc, params ); break;
case 285: gte_1hh( task_size, pc, params ); break;
case 286: gte_1ll( task_size, pc, params ); break;
case 287: gte_1qq( task_size, pc, params ); break;
case 288: gte_1BB( task_size, pc, params ); break;
case 289: gte_1HH( task_size, pc, params ); break;
case 290: gte_1LL( task_size, pc, params ); break;
case 291: gte_1QQ( task_size, pc, params ); break;
case 292: gte_1ff( task_size, pc, params ); break;
case 293: gte_1dd( task_size, pc, params ); break;
case 294: lt_111( task_size, pc, params ); break;
case 295: lt_1bb( task_size, pc, params ); break;
case 296: lt_1hh( task_size, pc, params ); break;
case 297: lt_1ll( task_size, pc, params ); break;
case 298: lt_1qq( task_size, pc, params ); break;
case 299: lt_1BB( task_size, pc, params ); break;
case 300: lt_1HH( task_size, pc, params ); break;
case 301: lt_1LL( task_size, pc, params ); break;
case 302: lt_1QQ( task_size, pc, params ); break;
case 303: lt_1ff( task_size, pc, params ); break;
case 304: lt_1dd( task_size, pc, params ); break;
case 305: lte_111( task_size, pc, params ); break;
case 306: lte_1bb( task_size, pc, params ); break;
case 307: lte_1hh( task_size, pc, params ); break;
case 308: lte_1ll( task_size, pc, params ); break;
case 309: lte_1qq( task_size, pc, params ); break;
case 310: lte_1BB( task_size, pc, params ); break;
case 311: lte_1HH( task_size, pc, params ); break;
case 312: lte_1LL( task_size, pc, params ); break;
case 313: lte_1QQ( task_size, pc, params ); break;
case 314: lte_1ff( task_size, pc, params ); break;
case 315: lte_1dd( task_size, pc, params ); break;
case 316: eq_111( task_size, pc, params ); break;
case 317: eq_1bb( task_size, pc, params ); break;
case 318: eq_1hh( task_size, pc, params ); break;
case 319: eq_1ll( task_size, pc, params ); break;
case 320: eq_1qq( task_size, pc, params ); break;
case 321: eq_1BB( task_size, pc, params ); break;
case 322: eq_1HH( task_size, pc, params ); break;
case 323: eq_1LL( task_size, pc, params ); break;
case 324: eq_1QQ( task_size, pc, params ); break;
case 325: eq_1ff( task_size, pc, params ); break;
case 326: eq_1dd( task_size, pc, params ); break;
case 327: noteq_111( task_size, pc, params ); break;
case 328: noteq_1bb( task_size, pc, params ); break;
case 329: noteq_1hh( task_size, pc, params ); break;
case 330: noteq_1ll( task_size, pc, params ); break;
case 331: noteq_1qq( task_size, pc, params ); break;
case 332: noteq_1BB( task_size, pc, params ); break;
case 333: noteq_1HH( task_size, pc, params ); break;
case 334: noteq_1LL( task_size, pc, params ); break;
case 335: noteq_1QQ( task_size, pc, params ); break;
case 336: noteq_1ff( task_size, pc, params ); break;
case 337: noteq_1dd( task_size, pc, params ); break;
case 338: abs_bb( task_size, pc, params ); break;
case 339: abs_hh( task_size, pc, params ); break;
case 340: abs_ll( task_size, pc, params ); break;
case 341: abs_qq( task_size, pc, params ); break;
case 342: abs_ff( task_size, pc, params ); break;
case 343: abs_dd( task_size, pc, params ); break;
case 344: arccos_ff( task_size, pc, params ); break;
case 345: arccos_dd( task_size, pc, params ); break;
case 346: arcsin_ff( task_size, pc, params ); break;
case 347: arcsin_dd( task_size, pc, params ); break;
case 348: arctan_ff( task_size, pc, params ); break;
case 349: arctan_dd( task_size, pc, params ); break;
case 350: arctan2_fff( task_size, pc, params ); break;
case 351: arctan2_ddd( task_size, pc, params ); break;
case 352: ceil_ff( task_size, pc, params ); break;
case 353: ceil_dd( task_size, pc, params ); break;
case 354: cos_ff( task_size, pc, params ); break;
case 355: cos_dd( task_size, pc, params ); break;
case 356: cosh_ff( task_size, pc, params ); break;
case 357: cosh_dd( task_size, pc, params ); break;
case 358: exp_ff( task_size, pc, params ); break;
case 359: exp_dd( task_size, pc, params ); break;
case 360: fabs_ff( task_size, pc, params ); break;
case 361: fabs_dd( task_size, pc, params ); break;
case 362: floor_ff( task_size, pc, params ); break;
case 363: floor_dd( task_size, pc, params ); break;
case 364: fmod_fff( task_size, pc, params ); break;
case 365: fmod_ddd( task_size, pc, params ); break;
case 366: log_ff( task_size, pc, params ); break;
case 367: log_dd( task_size, pc, params ); break;
case 368: log10_ff( task_size, pc, params ); break;
case 369: log10_dd( task_size, pc, params ); break;
case 370: sin_ff( task_size, pc, params ); break;
case 371: sin_dd( task_size, pc, params ); break;
case 372: sinh_ff( task_size, pc, params ); break;
case 373: sinh_dd( task_size, pc, params ); break;
case 374: sqrt_ff( task_size, pc, params ); break;
case 375: sqrt_dd( task_size, pc, params ); break;
case 376: tan_ff( task_size, pc, params ); break;
case 377: tan_dd( task_size, pc, params ); break;
case 378: tanh_ff( task_size, pc, params ); break;
case 379: tanh_dd( task_size, pc, params ); break;
case 380: fpclassify_lf( task_size, pc, params ); break;
case 381: fpclassify_ld( task_size, pc, params ); break;
case 382: isfinite_1f( task_size, pc, params ); break;
case 383: isfinite_1d( task_size, pc, params ); break;
case 384: isinf_1f( task_size, pc, params ); break;
case 385: isinf_1d( task_size, pc, params ); break;
case 386: isnan_1f( task_size, pc, params ); break;
case 387: isnan_1d( task_size, pc, params ); break;
case 388: isnormal_1f( task_size, pc, params ); break;
case 389: isnormal_1d( task_size, pc, params ); break;
case 390: signbit_1f( task_size, pc, params ); break;
case 391: signbit_1d( task_size, pc, params ); break;
case 392: arccosh_ff( task_size, pc, params ); break;
case 393: arccosh_dd( task_size, pc, params ); break;
case 394: arcsinh_ff( task_size, pc, params ); break;
case 395: arcsinh_dd( task_size, pc, params ); break;
case 396: arctanh_ff( task_size, pc, params ); break;
case 397: arctanh_dd( task_size, pc, params ); break;
case 398: cbrt_ff( task_size, pc, params ); break;
case 399: cbrt_dd( task_size, pc, params ); break;
case 400: copysign_fff( task_size, pc, params ); break;
case 401: copysign_ddd( task_size, pc, params ); break;
case 402: erf_ff( task_size, pc, params ); break;
case 403: erf_dd( task_size, pc, params ); break;
case 404: erfc_ff( task_size, pc, params ); break;
case 405: erfc_dd( task_size, pc, params ); break;
case 406: exp2_ff( task_size, pc, params ); break;
case 407: exp2_dd( task_size, pc, params ); break;
case 408: expm1_ff( task_size, pc, params ); break;
case 409: expm1_dd( task_size, pc, params ); break;
case 410: fdim_fff( task_size, pc, params ); break;
case 411: fdim_ddd( task_size, pc, params ); break;
case 412: fma_ffff( task_size, pc, params ); break;
case 413: fma_dddd( task_size, pc, params ); break;
case 414: fmax_fff( task_size, pc, params ); break;
case 415: fmax_ddd( task_size, pc, params ); break;
case 416: fmin_fff( task_size, pc, params ); break;
case 417: fmin_ddd( task_size, pc, params ); break;
case 418: hypot_fff( task_size, pc, params ); break;
case 419: hypot_ddd( task_size, pc, params ); break;
case 420: ilogb_lf( task_size, pc, params ); break;
case 421: ilogb_ld( task_size, pc, params ); break;
case 422: lgamma_ff( task_size, pc, params ); break;
case 423: lgamma_dd( task_size, pc, params ); break;
case 424: log1p_ff( task_size, pc, params ); break;
case 425: log1p_dd( task_size, pc, params ); break;
case 426: log2_ff( task_size, pc, params ); break;
case 427: log2_dd( task_size, pc, params ); break;
case 428: logb_ff( task_size, pc, params ); break;
case 429: logb_dd( task_size, pc, params ); break;
case 430: lrint_qf( task_size, pc, params ); break;
case 431: lrint_qd( task_size, pc, params ); break;
case 432: lround_qf( task_size, pc, params ); break;
case 433: lround_qd( task_size, pc, params ); break;
case 434: nearbyint_qf( task_size, pc, params ); break;
case 435: nearbyint_qd( task_size, pc, params ); break;
case 436: nextafter_fff( task_size, pc, params ); break;
case 437: nextafter_ddd( task_size, pc, params ); break;
case 438: nexttoward_fff( task_size, pc, params ); break;
case 439: nexttoward_ddd( task_size, pc, params ); break;
case 440: rint_ff( task_size, pc, params ); break;
case 441: rint_dd( task_size, pc, params ); break;
case 442: round_lf( task_size, pc, params ); break;
case 443: round_ld( task_size, pc, params ); break;
case 444: scalbln_ffq( task_size, pc, params ); break;
case 445: scalbln_ddq( task_size, pc, params ); break;
case 446: tgamma_ff( task_size, pc, params ); break;
case 447: tgamma_dd( task_size, pc, params ); break;
case 448: trunc_ff( task_size, pc, params ); break;
case 449: trunc_dd( task_size, pc, params ); break;
case 450: complex_Fff( task_size, pc, params ); break;
case 451: complex_Ddd( task_size, pc, params ); break;
case 452: real_fF( task_size, pc, params ); break;
case 453: real_dD( task_size, pc, params ); break;
case 454: imag_fF( task_size, pc, params ); break;
case 455: imag_dD( task_size, pc, params ); break;
case 456: abs_fF( task_size, pc, params ); break;
case 457: abs_dD( task_size, pc, params ); break;
case 458: abs2_fF( task_size, pc, params ); break;
case 459: abs2_dD( task_size, pc, params ); break;
case 460: add_FFF( task_size, pc, params ); break;
case 461: add_DDD( task_size, pc, params ); break;
case 462: sub_FFF( task_size, pc, params ); break;
case 463: sub_DDD( task_size, pc, params ); break;
case 464: mult_FFF( task_size, pc, params ); break;
case 465: mult_DDD( task_size, pc, params ); break;
case 466: div_FFF( task_size, pc, params ); break;
case 467: div_DDD( task_size, pc, params ); break;
case 468: usub_FF( task_size, pc, params ); break;
case 469: usub_DD( task_size, pc, params ); break;
case 470: neg_FF( task_size, pc, params ); break;
case 471: neg_DD( task_size, pc, params ); break;
case 472: conj_FF( task_size, pc, params ); break;
case 473: conj_DD( task_size, pc, params ); break;
case 474: conj_ff( task_size, pc, params ); break;
case 475: conj_dd( task_size, pc, params ); break;
case 476: sqrt_FF( task_size, pc, params ); break;
case 477: sqrt_DD( task_size, pc, params ); break;
case 478: log_FF( task_size, pc, params ); break;
case 479: log_DD( task_size, pc, params ); break;
case 480: log1p_FF( task_size, pc, params ); break;
case 481: log1p_DD( task_size, pc, params ); break;
case 482: log10_FF( task_size, pc, params ); break;
case 483: log10_DD( task_size, pc, params ); break;
case 484: exp_FF( task_size, pc, params ); break;
case 485: exp_DD( task_size, pc, params ); break;
case 486: expm1_FF( task_size, pc, params ); break;
case 487: expm1_DD( task_size, pc, params ); break;
case 488: pow_FFF( task_size, pc, params ); break;
case 489: pow_DDD( task_size, pc, params ); break;
case 490: arccos_FF( task_size, pc, params ); break;
case 491: arccos_DD( task_size, pc, params ); break;
case 492: arccosh_FF( task_size, pc, params ); break;
case 493: arccosh_DD( task_size, pc, params ); break;
case 494: arcsin_FF( task_size, pc, params ); break;
case 495: arcsin_DD( task_size, pc, params ); break;
case 496: arcsinh_FF( task_size, pc, params ); break;
case 497: arcsinh_DD( task_size, pc, params ); break;
case 498: arctan_FF( task_size, pc, params ); break;
case 499: arctan_DD( task_size, pc, params ); break;
case 500: arctanh_FF( task_size, pc, params ); break;
case 501: arctanh_DD( task_size, pc, params ); break;
case 502: cos_FF( task_size, pc, params ); break;
case 503: cos_DD( task_size, pc, params ); break;
case 504: cosh_FF( task_size, pc, params ); break;
case 505: cosh_DD( task_size, pc, params ); break;
case 506: sin_FF( task_size, pc, params ); break;
case 507: sin_DD( task_size, pc, params ); break;
case 508: sinh_FF( task_size, pc, params ); break;
case 509: sinh_DD( task_size, pc, params ); break;
case 510: tan_FF( task_size, pc, params ); break;
case 511: tan_DD( task_size, pc, params ); break;
case 512: tanh_FF( task_size, pc, params ); break;
case 513: tanh_DD( task_size, pc, params ); break;
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


