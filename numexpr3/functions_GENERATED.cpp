#include "numexpr_object.hpp"

static int
cast_b1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int8)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int8)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_h1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_i1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_l1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_B1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint8)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint8)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_H1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint16)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint16)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_I1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_L1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_f1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_d1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_F1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_D1( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_hb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_ib( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_lb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_fb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_db( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_Fb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_Db( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_ih( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_lh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_fh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_dh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_Fh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_Dh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_li( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_di( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_Di( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_dl( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_Dl( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_hB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_iB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_lB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_HB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint16)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint16)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_IB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_LB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_fB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_dB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_FB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_DB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_iH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_lH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_IH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_LH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_fH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_dH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_FH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_DH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_lI( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_LI( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_dI( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_DI( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_dL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_DL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_df( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    return 0;
    }


static int
cast_Ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float32)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float32)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_Df( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_Dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]); dest[J].imag=0.0; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]); dest[J].imag=0.0; 
    }
    return 0;
    }


static int
cast_DF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_complex64 *x1 = (npy_complex64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J].real = (npy_float64)(x1[J]).real; dest[J].imag=(npy_float64)(x1[J]).imag; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_complex64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J].real = (npy_float64)(x1[J*sb1]).real; dest[J].imag=(npy_float64)(x1[J*sb1]).imag; 
    }
    return 0;
    }


static int
copy_11( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_complex64 *x1 = (npy_complex64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_complex64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
copy_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_complex128 *x1 = (npy_complex128 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_complex128);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1]; 
    }
    return 0;
    }


static int
add_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
add_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
sub_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
mult_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    return 0;
    }


static int
div_d11( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dII( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_dLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (npy_float64)x1[J] / (npy_float64)x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (npy_float64)x1[J*sb1] / (npy_float64)x2[J*sb2]; 
    }
    return 0;
    }


static int
div_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] / x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] / x2[J*sb2]; 
    }
    return 0;
    }


static int
div_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] / x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] / x2[J*sb2]; 
    }
    return 0;
    }


static int
pow_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
pow_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
mod_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    return 0;
    }


static int
mod_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    return 0;
    }


static int
where_1111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_bool *x3 = (npy_bool *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) && sb3 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_b1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_int8 *x3 = (npy_int8 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_int8) && sb3 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_h1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_int16 *x3 = (npy_int16 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_int16) && sb3 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_i1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_int32 *x3 = (npy_int32 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_int32) && sb3 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_l1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_int64 *x3 = (npy_int64 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_int64) && sb3 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_B1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_uint8 *x3 = (npy_uint8 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_uint8) && sb3 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_H1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_uint16 *x3 = (npy_uint16 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_uint16) && sb3 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_I1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_uint32 *x3 = (npy_uint32 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_uint32) && sb3 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_L1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_uint64 *x3 = (npy_uint64 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_uint64) && sb3 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_f1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_float32 *x3 = (npy_float32 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_float32) && sb3 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_d1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_float64 *x3 = (npy_float64 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_float64) && sb3 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_F1FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_complex64 *x2 = (npy_complex64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_complex64 *x3 = (npy_complex64 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_complex64) && sb3 == sizeof(npy_complex64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_complex64);
    sb2 /= sizeof(npy_complex64);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
where_D1DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_complex128 *x2 = (npy_complex128 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_complex128 *x3 = (npy_complex128 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_complex128) && sb3 == sizeof(npy_complex128) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_complex128);
    sb2 /= sizeof(npy_complex128);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    return 0;
    }


static int
ones_like_11( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
ones_like_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = 1; 
}
        return 0;
    } 
    // Strided
    for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = 1; 
    }
    return 0;
    }


static int
usub_bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
usub_hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
usub_ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
usub_ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
usub_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
usub_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    return 0;
    }


static int
lshift_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
lshift_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
rshift_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    return 0;
    }


static int
bitand_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitand_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitor_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_bbb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_hhh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_iii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_lll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_BBB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_HHH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_III( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
bitxor_LLL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    return 0;
    }


static int
logical_and_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] && x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] && x2[J*sb2]); 
    }
    return 0;
    }


static int
logical_or_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] || x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] || x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gt_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
gte_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lt_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
lte_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
eq_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_111( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_bool *x2 = (npy_bool *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_bool) && sb2 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_bool);
    sb1 /= sizeof(npy_bool);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int8 *x2 = (npy_int8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int8) && sb2 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int8);
    sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int16 *x2 = (npy_int16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int16) && sb2 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int16);
    sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int32);
    sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_int64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1BB( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint8 *x2 = (npy_uint8 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint8) && sb2 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint8);
    sb1 /= sizeof(npy_uint8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1HH( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint16 *x2 = (npy_uint16 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint16) && sb2 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint16);
    sb1 /= sizeof(npy_uint16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1II( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint32 *x2 = (npy_uint32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint32) && sb2 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint32);
    sb1 /= sizeof(npy_uint32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1LL( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_uint64 *x2 = (npy_uint64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_uint64) && sb2 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_uint64);
    sb1 /= sizeof(npy_uint64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
noteq_1dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    return 0;
    }


static int
abs_bb( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int8);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
abs_hh( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int16);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
abs_ii( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
abs_ll( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_int64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
abs_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
abs_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = x1[J] < 0 ? -x1[J] : x1[J]; 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = x1[J*sb1] < 0 ? -x1[J*sb1] : x1[J*sb1]; 
    }
    return 0;
    }


static int
arccos_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = acos(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = acos(x1[J*sb1]); 
    }
    return 0;
    }


static int
arccos_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = acos(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = acos(x1[J*sb1]); 
    }
    return 0;
    }


static int
arcsin_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = asin(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = asin(x1[J*sb1]); 
    }
    return 0;
    }


static int
arcsin_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = asin(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = asin(x1[J*sb1]); 
    }
    return 0;
    }


static int
arctan_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atan(x1[J*sb1]); 
    }
    return 0;
    }


static int
arctan_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atan(x1[J*sb1]); 
    }
    return 0;
    }


static int
arctan2_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atan2(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atan2(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
arctan2_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atan2(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atan2(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
ceil_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = ceil(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = ceil(x1[J*sb1]); 
    }
    return 0;
    }


static int
ceil_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = ceil(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = ceil(x1[J*sb1]); 
    }
    return 0;
    }


static int
cos_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cos(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cos(x1[J*sb1]); 
    }
    return 0;
    }


static int
cos_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cos(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cos(x1[J*sb1]); 
    }
    return 0;
    }


static int
cosh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cosh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cosh(x1[J*sb1]); 
    }
    return 0;
    }


static int
cosh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cosh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cosh(x1[J*sb1]); 
    }
    return 0;
    }


static int
exp_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = exp(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = exp(x1[J*sb1]); 
    }
    return 0;
    }


static int
exp_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = exp(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = exp(x1[J*sb1]); 
    }
    return 0;
    }


static int
fabs_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fabs(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fabs(x1[J*sb1]); 
    }
    return 0;
    }


static int
fabs_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fabs(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fabs(x1[J*sb1]); 
    }
    return 0;
    }


static int
floor_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = floor(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = floor(x1[J*sb1]); 
    }
    return 0;
    }


static int
floor_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = floor(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = floor(x1[J*sb1]); 
    }
    return 0;
    }


static int
fmod_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmod(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmod(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fmod_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmod(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmod(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
log_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log(x1[J*sb1]); 
    }
    return 0;
    }


static int
log_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log(x1[J*sb1]); 
    }
    return 0;
    }


static int
log10_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log10(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log10(x1[J*sb1]); 
    }
    return 0;
    }


static int
log10_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log10(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log10(x1[J*sb1]); 
    }
    return 0;
    }


static int
sin_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sin(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sin(x1[J*sb1]); 
    }
    return 0;
    }


static int
sin_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sin(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sin(x1[J*sb1]); 
    }
    return 0;
    }


static int
sinh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sinh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sinh(x1[J*sb1]); 
    }
    return 0;
    }


static int
sinh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sinh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sinh(x1[J*sb1]); 
    }
    return 0;
    }


static int
sqrt_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sqrt(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sqrt(x1[J*sb1]); 
    }
    return 0;
    }


static int
sqrt_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = sqrt(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = sqrt(x1[J*sb1]); 
    }
    return 0;
    }


static int
tan_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tan(x1[J*sb1]); 
    }
    return 0;
    }


static int
tan_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tan(x1[J*sb1]); 
    }
    return 0;
    }


static int
tanh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tanh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tanh(x1[J*sb1]); 
    }
    return 0;
    }


static int
tanh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tanh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tanh(x1[J*sb1]); 
    }
    return 0;
    }


static int
fpclassify_if( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fpclassify(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fpclassify(x1[J*sb1]); 
    }
    return 0;
    }


static int
fpclassify_id( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fpclassify(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fpclassify(x1[J*sb1]); 
    }
    return 0;
    }


static int
isfinite_1f( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isfinite(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isfinite(x1[J*sb1]); 
    }
    return 0;
    }


static int
isfinite_1d( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isfinite(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isfinite(x1[J*sb1]); 
    }
    return 0;
    }


static int
isinf_1f( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isinf(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isinf(x1[J*sb1]); 
    }
    return 0;
    }


static int
isinf_1d( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isinf(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isinf(x1[J*sb1]); 
    }
    return 0;
    }


static int
isnan_1f( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isnan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isnan(x1[J*sb1]); 
    }
    return 0;
    }


static int
isnan_1d( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isnan(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isnan(x1[J*sb1]); 
    }
    return 0;
    }


static int
isnormal_1f( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isnormal(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isnormal(x1[J*sb1]); 
    }
    return 0;
    }


static int
isnormal_1d( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = isnormal(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = isnormal(x1[J*sb1]); 
    }
    return 0;
    }


static int
signbit_1f( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = signbit(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = signbit(x1[J*sb1]); 
    }
    return 0;
    }


static int
signbit_1d( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = signbit(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = signbit(x1[J*sb1]); 
    }
    return 0;
    }


static int
arccosh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = acosh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = acosh(x1[J*sb1]); 
    }
    return 0;
    }


static int
arccosh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = acosh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = acosh(x1[J*sb1]); 
    }
    return 0;
    }


static int
arcsinh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = asinh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = asinh(x1[J*sb1]); 
    }
    return 0;
    }


static int
arcsinh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = asinh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = asinh(x1[J*sb1]); 
    }
    return 0;
    }


static int
arctanh_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atanh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atanh(x1[J*sb1]); 
    }
    return 0;
    }


static int
arctanh_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = atanh(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = atanh(x1[J*sb1]); 
    }
    return 0;
    }


static int
cbrt_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cbrt(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cbrt(x1[J*sb1]); 
    }
    return 0;
    }


static int
cbrt_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = cbrt(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = cbrt(x1[J*sb1]); 
    }
    return 0;
    }


static int
copysign_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = copysign(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = copysign(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
copysign_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = copysign(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = copysign(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
erf_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = erf(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = erf(x1[J*sb1]); 
    }
    return 0;
    }


static int
erf_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = erf(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = erf(x1[J*sb1]); 
    }
    return 0;
    }


static int
erfc_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = erfc(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = erfc(x1[J*sb1]); 
    }
    return 0;
    }


static int
erfc_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = erfc(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = erfc(x1[J*sb1]); 
    }
    return 0;
    }


static int
exp2_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = exp2(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = exp2(x1[J*sb1]); 
    }
    return 0;
    }


static int
exp2_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = exp2(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = exp2(x1[J*sb1]); 
    }
    return 0;
    }


static int
expm1_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = expm1(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = expm1(x1[J*sb1]); 
    }
    return 0;
    }


static int
expm1_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = expm1(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = expm1(x1[J*sb1]); 
    }
    return 0;
    }


static int
fdim_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fdim(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fdim(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fdim_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fdim(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fdim(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fma_ffff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_float32 *x3 = (npy_float32 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) && sb3 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fma(x1[J], x2[J], x3[J]); 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fma(x1[J*sb1], x2[J*sb2], x3[J*sb3]); 
    }
    return 0;
    }


static int
fma_dddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    NE_REGISTER arg3 = params->program[pc].arg3;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    BOUNDS_CHECK(arg3);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
    npy_float64 *x3 = (npy_float64 *)params->registers[arg3].mem;
    npy_intp sb3 = params->registers[arg3].stride;
                                
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) && sb3 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fma(x1[J], x2[J], x3[J]); 
}
        return 0;
    }
    // Strided
        sb3 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fma(x1[J*sb1], x2[J*sb2], x3[J*sb3]); 
    }
    return 0;
    }


static int
fmax_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmax(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmax(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fmax_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmax(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmax(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fmin_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmin(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmin(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
fmin_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = fmin(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = fmin(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
hypot_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = hypot(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = hypot(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
hypot_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = hypot(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = hypot(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
ilogb_if( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = ilogb(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = ilogb(x1[J*sb1]); 
    }
    return 0;
    }


static int
ilogb_id( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = ilogb(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = ilogb(x1[J*sb1]); 
    }
    return 0;
    }


static int
lgamma_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lgamma(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lgamma(x1[J*sb1]); 
    }
    return 0;
    }


static int
lgamma_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lgamma(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lgamma(x1[J*sb1]); 
    }
    return 0;
    }


static int
log1p_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log1p(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log1p(x1[J*sb1]); 
    }
    return 0;
    }


static int
log1p_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log1p(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log1p(x1[J*sb1]); 
    }
    return 0;
    }


static int
log2_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log2(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log2(x1[J*sb1]); 
    }
    return 0;
    }


static int
log2_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = log2(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = log2(x1[J*sb1]); 
    }
    return 0;
    }


static int
logb_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = logb(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = logb(x1[J*sb1]); 
    }
    return 0;
    }


static int
logb_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = logb(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = logb(x1[J*sb1]); 
    }
    return 0;
    }


static int
lrint_lf( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lrint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lrint(x1[J*sb1]); 
    }
    return 0;
    }


static int
lrint_ld( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lrint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lrint(x1[J*sb1]); 
    }
    return 0;
    }


static int
lround_lf( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lround(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lround(x1[J*sb1]); 
    }
    return 0;
    }


static int
lround_ld( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = lround(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = lround(x1[J*sb1]); 
    }
    return 0;
    }


static int
nearbyint_lf( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nearbyint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nearbyint(x1[J*sb1]); 
    }
    return 0;
    }


static int
nearbyint_ld( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nearbyint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nearbyint(x1[J*sb1]); 
    }
    return 0;
    }


static int
nextafter_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nextafter(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nextafter(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
nextafter_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nextafter(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nextafter(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
nexttoward_fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float32 *x2 = (npy_float32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nexttoward(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float32);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nexttoward(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
nexttoward_ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_float64 *x2 = (npy_float64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = nexttoward(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_float64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = nexttoward(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
rint_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = rint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = rint(x1[J*sb1]); 
    }
    return 0;
    }


static int
rint_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = rint(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = rint(x1[J*sb1]); 
    }
    return 0;
    }


static int
round_if( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = round(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = round(x1[J*sb1]); 
    }
    return 0;
    }


static int
round_id( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = round(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = round(x1[J*sb1]); 
    }
    return 0;
    }


static int
scalbln_ffl( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = scalbln(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = scalbln(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
scalbln_ddl( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    NE_REGISTER arg2 = params->program[pc].arg2;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);
    BOUNDS_CHECK(arg2);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
    npy_int64 *x2 = (npy_int64 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = scalbln(x1[J], x2[J]); 
}
        return 0;
    }
    // Strided
        sb2 /= sizeof(npy_int64);
    sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = scalbln(x1[J*sb1], x2[J*sb2]); 
    }
    return 0;
    }


static int
tgamma_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tgamma(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tgamma(x1[J*sb1]); 
    }
    return 0;
    }


static int
tgamma_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = tgamma(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = tgamma(x1[J*sb1]); 
    }
    return 0;
    }


static int
trunc_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = trunc(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float32);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = trunc(x1[J*sb1]); 
    }
    return 0;
    }


static int
trunc_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < block_size; J++) { 
    dest[J] = trunc(x1[J]); 
}
        return 0;
    } 
    // Strided
        sb1 /= sizeof(npy_float64);
for(npy_intp J = 0; J < block_size; J++) { 
        dest[J] = trunc(x1[J*sb1]); 
    }
    return 0;
    }


static int
complex_Fff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_complex(block_size, (npy_float32 *)x1, (npy_float32 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
complex_Ddd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_complex(block_size, (npy_float64 *)x1, (npy_float64 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
real_fF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_real(block_size, (npy_complex64 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
real_dD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_real(block_size, (npy_complex128 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
imag_fF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_imag(block_size, (npy_complex64 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
imag_dD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_imag(block_size, (npy_complex128 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
abs_fF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs(block_size, (npy_complex64 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
abs_dD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs(block_size, (npy_complex128 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
abs2_fF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs2(block_size, (npy_complex64 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
abs2_dD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs2(block_size, (npy_complex128 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
add_FFF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_add(block_size, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
add_DDD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_add(block_size, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
sub_FFF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_sub(block_size, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
sub_DDD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_sub(block_size, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
mult_FFF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_mul(block_size, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
mult_DDD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_mul(block_size, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
div_FFF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_div(block_size, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
div_DDD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_div(block_size, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
usub_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
usub_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
neg_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
neg_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
conj_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_conj(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
conj_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_conj(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
conj_ff( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        fconj(block_size, (npy_float32 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
conj_dd( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        fconj(block_size, (npy_float64 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
sqrt_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sqrt(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sqrt_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sqrt(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log1p_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log1p(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log1p_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log1p(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log10_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log10(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log10_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log10(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
exp_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_exp(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
exp_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_exp(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
expm1_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_expm1(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
expm1_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_expm1(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
pow_FFF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_pow(block_size, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
pow_DDD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        char *x2 = params->registers[arg2].mem;
        nc_pow(block_size, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
arccos_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acos(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arccos_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acos(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arccosh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acosh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arccosh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acosh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arcsin_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asin(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arcsin_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asin(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arcsinh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asinh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arcsinh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asinh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arctan_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atan(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arctan_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atan(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arctanh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atanh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arctanh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atanh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
cos_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cos(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
cos_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cos(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
cosh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cosh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
cosh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cosh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
sin_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sin(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sin_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sin(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
sinh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sinh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sinh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sinh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
tan_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tan(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
tan_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tan(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
tanh_FF( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tanh(block_size, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
tanh_DD( npy_intp block_size, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tanh(block_size, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


