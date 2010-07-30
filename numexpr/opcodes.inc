/*
OPCODE(n, enum_name, exported, return_type, arg1_type, arg2_type, arg3_type)

`exported` is NULL if the opcode shouldn't exported by the Python module.

Types are Tb, Ti, Tl, Tf, Td, Tc, Ts, Tn, and T0; these symbols should be
#defined to whatever is needed. (T0 is the no-such-arg type.)

*/
OPCODE(0, OP_NOOP, "noop", T0, T0, T0, T0)

OPCODE(1, OP_COPY_BB, "copy_bb", Tb, Tb, T0, T0)

OPCODE(2, OP_INVERT_BB, "invert_bb", Tb, Tb, T0, T0)
OPCODE(3, OP_AND_BBB, "and_bbb", Tb, Tb, Tb, T0)
OPCODE(4, OP_OR_BBB, "or_bbb", Tb, Tb, Tb, T0)

OPCODE(5, OP_EQ_BBB, "eq_bbb", Tb, Tb, Tb, T0)
OPCODE(6, OP_NE_BBB, "ne_bbb", Tb, Tb, Tb, T0)

OPCODE(7, OP_GT_BII, "gt_bii", Tb, Ti, Ti, T0)
OPCODE(8, OP_GE_BII, "ge_bii", Tb, Ti, Ti, T0)
OPCODE(9, OP_EQ_BII, "eq_bii", Tb, Ti, Ti, T0)
OPCODE(10, OP_NE_BII, "ne_bii", Tb, Ti, Ti, T0)

OPCODE(11, OP_GT_BLL, "gt_bll", Tb, Tl, Tl, T0)
OPCODE(12, OP_GE_BLL, "ge_bll", Tb, Tl, Tl, T0)
OPCODE(13, OP_EQ_BLL, "eq_bll", Tb, Tl, Tl, T0)
OPCODE(14, OP_NE_BLL, "ne_bll", Tb, Tl, Tl, T0)

OPCODE(15, OP_GT_BFF, "gt_bff", Tb, Tf, Tf, T0)
OPCODE(16, OP_GE_BFF, "ge_bff", Tb, Tf, Tf, T0)
OPCODE(17, OP_EQ_BFF, "eq_bff", Tb, Tf, Tf, T0)
OPCODE(18, OP_NE_BFF, "ne_bff", Tb, Tf, Tf, T0)

OPCODE(19, OP_GT_BDD, "gt_bdd", Tb, Td, Td, T0)
OPCODE(20, OP_GE_BDD, "ge_bdd", Tb, Td, Td, T0)
OPCODE(21, OP_EQ_BDD, "eq_bdd", Tb, Td, Td, T0)
OPCODE(22, OP_NE_BDD, "ne_bdd", Tb, Td, Td, T0)

OPCODE(23, OP_GT_BSS, "gt_bss", Tb, Ts, Ts, T0)
OPCODE(24, OP_GE_BSS, "ge_bss", Tb, Ts, Ts, T0)
OPCODE(25, OP_EQ_BSS, "eq_bss", Tb, Ts, Ts, T0)
OPCODE(26, OP_NE_BSS, "ne_bss", Tb, Ts, Ts, T0)

OPCODE(27, OP_CAST_IB, "cast_ib", Ti, Tb, T0, T0)
OPCODE(28, OP_COPY_II, "copy_ii", Ti, Ti, T0, T0)
OPCODE(29, OP_ONES_LIKE_I, "ones_like_i", Ti, T0, T0, T0)
OPCODE(30, OP_NEG_II, "neg_ii", Ti, Ti, T0, T0)
OPCODE(31, OP_ADD_III, "add_iii", Ti, Ti, Ti, T0)
OPCODE(32, OP_SUB_III, "sub_iii", Ti, Ti, Ti, T0)
OPCODE(33, OP_MUL_III, "mul_iii", Ti, Ti, Ti, T0)
OPCODE(34, OP_DIV_III, "div_iii", Ti, Ti, Ti, T0)
OPCODE(35, OP_POW_III, "pow_iii", Ti, Ti, Ti, T0)
OPCODE(36, OP_MOD_III, "mod_iii", Ti, Ti, Ti, T0)
OPCODE(37, OP_WHERE_IBII, "where_ibii", Ti, Tb, Ti, Ti)

OPCODE(38, OP_CAST_LI, "cast_li", Tl, Ti, T0, T0)
OPCODE(39, OP_COPY_LL, "copy_ll", Tl, Tl, T0, T0)
OPCODE(40, OP_ONES_LIKE_L, "ones_like_l", Tl, T0, T0, T0)
OPCODE(41, OP_NEG_LL, "neg_ll", Tl, Tl, T0, T0)
OPCODE(42, OP_ADD_LLL, "add_lll", Tl, Tl, Tl, T0)
OPCODE(43, OP_SUB_LLL, "sub_lll", Tl, Tl, Tl, T0)
OPCODE(44, OP_MUL_LLL, "mul_lll", Tl, Tl, Tl, T0)
OPCODE(45, OP_DIV_LLL, "div_lll", Tl, Tl, Tl, T0)
OPCODE(46, OP_POW_LLL, "pow_lll", Tl, Tl, Tl, T0)
OPCODE(47, OP_MOD_LLL, "mod_lll", Tl, Tl, Tl, T0)
OPCODE(48, OP_WHERE_LBLL, "where_lbll", Tl, Tb, Tl, Tl)

OPCODE(49, OP_CAST_FI, "cast_fi", Tf, Ti, T0, T0)
OPCODE(50, OP_CAST_FL, "cast_fl", Tf, Tl, T0, T0)
OPCODE(51, OP_COPY_FF, "copy_ff", Tf, Tf, T0, T0)
OPCODE(52, OP_ONES_LIKE_F, "ones_like_f", Tf, T0, T0, T0)
OPCODE(53, OP_NEG_FF, "neg_ff", Tf, Tf, T0, T0)
OPCODE(54, OP_ADD_FFF, "add_fff", Tf, Tf, Tf, T0)
OPCODE(55, OP_SUB_FFF, "sub_fff", Tf, Tf, Tf, T0)
OPCODE(56, OP_MUL_FFF, "mul_fff", Tf, Tf, Tf, T0)
OPCODE(57, OP_DIV_FFF, "div_fff", Tf, Tf, Tf, T0)
OPCODE(58, OP_POW_FFF, "pow_fff", Tf, Tf, Tf, T0)
OPCODE(59, OP_MOD_FFF, "mod_fff", Tf, Tf, Tf, T0)
OPCODE(60, OP_SQRT_FF, "sqrt_ff", Tf, Tf, T0, T0)
OPCODE(61, OP_WHERE_FBFF, "where_fbff", Tf, Tb, Tf, Tf)
OPCODE(62, OP_FUNC_FFN, "func_ffn", Tf, Tf, Tn, T0)
OPCODE(63, OP_FUNC_FFFN, "func_fffn", Tf, Tf, Tf, Tn)

OPCODE(64, OP_CAST_DI, "cast_di", Td, Ti, T0, T0)
OPCODE(65, OP_CAST_DL, "cast_dl", Td, Tl, T0, T0)
OPCODE(66, OP_CAST_DF, "cast_df", Td, Tf, T0, T0)
OPCODE(67, OP_COPY_DD, "copy_dd", Td, Td, T0, T0)
OPCODE(68, OP_ONES_LIKE_D, "ones_like_d", Td, T0, T0, T0)
OPCODE(69, OP_NEG_DD, "neg_dd", Td, Td, T0, T0)
OPCODE(70, OP_ADD_DDD, "add_ddd", Td, Td, Td, T0)
OPCODE(71, OP_SUB_DDD, "sub_ddd", Td, Td, Td, T0)
OPCODE(72, OP_MUL_DDD, "mul_ddd", Td, Td, Td, T0)
OPCODE(73, OP_DIV_DDD, "div_ddd", Td, Td, Td, T0)
OPCODE(74, OP_POW_DDD, "pow_ddd", Td, Td, Td, T0)
OPCODE(75, OP_MOD_DDD, "mod_ddd", Td, Td, Td, T0)
OPCODE(76, OP_SQRT_DD, "sqrt_dd", Td, Td, T0, T0)
OPCODE(77, OP_WHERE_DBDD, "where_dbdd", Td, Tb, Td, Td)
OPCODE(78, OP_FUNC_DDN, "func_ddn", Td, Td, Tn, T0)
OPCODE(79, OP_FUNC_DDDN, "func_dddn", Td, Td, Td, Tn)

OPCODE(80, OP_EQ_BCC, "eq_bcc", Tb, Tc, Tc, T0)
OPCODE(81, OP_NE_BCC, "ne_bcc", Tb, Tc, Tc, T0)

OPCODE(82, OP_CAST_CI, "cast_ci", Tc, Ti, T0, T0)
OPCODE(83, OP_CAST_CL, "cast_cl", Tc, Tl, T0, T0)
OPCODE(84, OP_CAST_CF, "cast_cf", Tc, Tf, T0, T0)
OPCODE(85, OP_CAST_CD, "cast_cd", Tc, Td, T0, T0)
OPCODE(86, OP_ONES_LIKE_C, "ones_like_c", Tc, T0, T0, T0)
OPCODE(87, OP_COPY_CC, "copy_cc", Tc, Tc, T0, T0)
OPCODE(88, OP_NEG_CC, "neg_cc", Tc, Tc, T0, T0)
OPCODE(89, OP_ADD_CCC, "add_ccc", Tc, Tc, Tc, T0)
OPCODE(90, OP_SUB_CCC, "sub_ccc", Tc, Tc, Tc, T0)
OPCODE(91, OP_MUL_CCC, "mul_ccc", Tc, Tc, Tc, T0)
OPCODE(92, OP_DIV_CCC, "div_ccc", Tc, Tc, Tc, T0)
OPCODE(93, OP_WHERE_CBCC, "where_cbcc", Tc, Tb, Tc, Tc)
OPCODE(94, OP_FUNC_CCN, "func_ccn", Tc, Tc, Tn, T0)
OPCODE(95, OP_FUNC_CCCN, "func_cccn", Tc, Tc, Tc, Tn)

OPCODE(96, OP_REAL_DC, "real_dc", Td, Tc, T0, T0)
OPCODE(97, OP_IMAG_DC, "imag_dc", Td, Tc, T0, T0)
OPCODE(98, OP_COMPLEX_CDD, "complex_cdd", Tc, Td, Td, T0)

OPCODE(99, OP_COPY_SS, "copy_ss", Ts, Ts, T0, T0)

OPCODE(100, OP_WHERE_BBBB, "where_bbbb", Tb, Tb, Tb, Tb)

OPCODE(101, OP_REDUCTION, NULL, T0, T0, T0, T0)

/* Last argument in a reduction is the axis of the array the
   reduction should be applied along. */

OPCODE(102, OP_SUM, NULL, T0, T0, T0, T0)
OPCODE(103, OP_SUM_IIN, "sum_iin", Ti, Ti, Tn, T0)
OPCODE(104, OP_SUM_LLN, "sum_lln", Tl, Tl, Tn, T0)
OPCODE(105, OP_SUM_FFN, "sum_ffn", Tf, Tf, Tn, T0)
OPCODE(106, OP_SUM_DDN, "sum_ddn", Td, Td, Tn, T0)
OPCODE(107, OP_SUM_CCN, "sum_ccn", Tc, Tc, Tn, T0)

OPCODE(108, OP_PROD, NULL, T0, T0, T0, T0)
OPCODE(109, OP_PROD_IIN, "prod_iin", Ti, Ti, Tn, T0)
OPCODE(110, OP_PROD_LLN, "prod_lln", Tl, Tl, Tn, T0)
OPCODE(111, OP_PROD_FFN, "prod_ffn", Tf, Tf, Tn, T0)
OPCODE(112, OP_PROD_DDN, "prod_ddn", Td, Td, Tn, T0)
OPCODE(113, OP_PROD_CCN, "prod_ccn", Tc, Tc, Tn, T0)

/* Should be the last opcode */
OPCODE(114, OP_END, NULL, T0, T0, T0, T0)
