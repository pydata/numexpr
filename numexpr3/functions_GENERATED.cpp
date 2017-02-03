#include "numexpr_object.hpp"

static int
cast_11( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_bool)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_bool)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_b1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int8)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int8)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_h1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_i1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_l1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_B1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint8)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint8)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_H1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_I1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_L1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_f1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_d1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_bb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int8)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int8)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_hb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_ib( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_lb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_fb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_db( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_hh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_ih( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_lh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_fh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_ii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_li( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_di( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_ll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dl( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_hB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_iB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_lB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_BB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint8)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint8)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_HB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_IB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_LB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_fB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_iH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_lH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_HH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint16)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint16)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_IH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_LH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_fH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_lI( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_int64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_int64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_II( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_LI( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dI( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_LL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_uint64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_uint64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float32)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float32)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_df( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cast_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (npy_float64)(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (npy_float64)(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
copy_11( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_bool *x1 = (npy_bool *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_bool) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_bool)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_bool)); 
    }
    }
    return 0;
    }


static int
copy_bb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int8)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int8)); 
    }
    }
    return 0;
    }


static int
copy_hh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int16)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int16)); 
    }
    }
    return 0;
    }


static int
copy_ii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int32)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int32)); 
    }
    }
    return 0;
    }


static int
copy_ll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int64)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_int64)); 
    }
    }
    return 0;
    }


static int
copy_BB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    npy_uint8 *x1 = (npy_uint8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint8)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint8)); 
    }
    }
    return 0;
    }


static int
copy_HH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    npy_uint16 *x1 = (npy_uint16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint16)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint16)); 
    }
    }
    return 0;
    }


static int
copy_II( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    npy_uint32 *x1 = (npy_uint32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint32)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint32)); 
    }
    }
    return 0;
    }


static int
copy_LL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    npy_uint64 *x1 = (npy_uint64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_uint64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint64)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_uint64)); 
    }
    }
    return 0;
    }


static int
copy_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_float32)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_float32)); 
    }
    }
    return 0;
    }


static int
copy_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_float64)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_float64)); 
    }
    }
    return 0;
    }


static int
copy_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex64 *dest = (npy_complex64 *)params->registers[store_in].mem;
    npy_complex64 *x1 = (npy_complex64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_complex64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_complex64)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_complex64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_complex64)); 
    }
    }
    return 0;
    }


static int
copy_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_complex128 *dest = (npy_complex128 *)params->registers[store_in].mem;
    npy_complex128 *x1 = (npy_complex128 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_complex128) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_complex128)); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_complex128);
    for(npy_intp J = 0; J < blocksize; J++) { 
        memcpy(&dest[J], ((char *)x1+J*sb1), sizeof(npy_complex128)); 
    }
    }
    return 0;
    }


static int
add_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
add_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] + x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] + x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
sub_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mult_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
div_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
div_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x2[J] ? (x1[J] / x2[J]) : 0; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x2[J*sb2] ? (x1[J*sb1] / x2[J*sb2]) : 0; 
    }
    }
    return 0;
    }


static int
pow_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
pow_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
mod_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
mod_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] - floor(x1[J]/x2[J]) * x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] - floor(x1[J*sb1]/x2[J*sb2]) * x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
where_1111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_b1bb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_h1hh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_i1ii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_l1ll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_B1BB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_H1HH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_I1II( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_L1LL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_f1ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
where_d1dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] ? x2[J] : x3[J]; 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    sb3 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] ? x2[J*sb2] : x3[J*sb3]; 
    }
    }
    return 0;
    }


static int
ones_like_1( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_b( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_h( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_i( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_l( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_B( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_uint8 *dest = (npy_uint8 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_H( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_uint16 *dest = (npy_uint16 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_I( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_uint32 *dest = (npy_uint32 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_L( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_uint64 *dest = (npy_uint64 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
ones_like_d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
    NE_REGISTER store_in = params->program[pc].ret;
    BOUNDS_CHECK(store_in);
    
    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    
    for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = 1; 
}
    return 0;
    }


static int
neg_bb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int8 *dest = (npy_int8 *)params->registers[store_in].mem;
    npy_int8 *x1 = (npy_int8 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int8) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
neg_hh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int16 *dest = (npy_int16 *)params->registers[store_in].mem;
    npy_int16 *x1 = (npy_int16 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int16) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
neg_ii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_int32 *x1 = (npy_int32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
neg_ll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_int64 *x1 = (npy_int64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_int64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
neg_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
neg_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = -x1[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = -x1[J*sb1]; 
    }
    }
    return 0;
    }


static int
lshift_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
lshift_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] << x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] << x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
rshift_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = x1[J] >> x2[J]; 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = x1[J*sb1] >> x2[J*sb2]; 
    }
    }
    return 0;
    }


static int
bitand_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitand_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] & x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] & x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitor_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] | x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] | x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
bitxor_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] ^ x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] ^ x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
and_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] && x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] && x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
or_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] || x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] || x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gt_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] > x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] > x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
gte_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] >= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] >= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lt_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] < x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] < x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
lte_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] <= x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] <= x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
eq_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] == x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] == x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_111( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_bool);
    sb2 /= sizeof(npy_bool);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_bbb( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int8);
    sb2 /= sizeof(npy_int8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_hhh( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int16);
    sb2 /= sizeof(npy_int16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_iii( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int32);
    sb2 /= sizeof(npy_int32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_lll( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_int64);
    sb2 /= sizeof(npy_int64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_BBB( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint8);
    sb2 /= sizeof(npy_uint8);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_HHH( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint16);
    sb2 /= sizeof(npy_uint16);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_III( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint32);
    sb2 /= sizeof(npy_uint32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_LLL( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_uint64);
    sb2 /= sizeof(npy_uint64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
noteq_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = (x1[J] != x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = (x1[J*sb1] != x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
abs_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = abs(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = abs(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
abs_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = abs(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = abs(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arccos_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = acos(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = acos(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arccos_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = acos(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = acos(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arcsin_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = asin(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = asin(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arcsin_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = asin(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = asin(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arctan_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arctan_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arctan2_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atan2(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atan2(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
arctan2_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atan2(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atan2(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
ceil_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = ceil(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = ceil(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
ceil_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = ceil(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = ceil(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cos_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cos(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cos(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cos_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cos(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cos(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cosh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cosh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cosh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cosh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cosh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cosh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
exp_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = exp(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = exp(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
exp_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = exp(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = exp(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fabs_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fabs(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fabs(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fabs_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fabs(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fabs(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
floor_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = floor(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = floor(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
floor_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = floor(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = floor(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fmod_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmod(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmod(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fmod_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmod(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmod(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
log_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log10_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log10(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log10(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log10_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log10(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log10(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
power_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
power_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
power_ffi( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float32) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
power_ddi( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
    npy_int32 *x2 = (npy_int32 *)params->registers[arg2].mem;
    npy_intp sb2 = params->registers[arg2].stride;
                                    
    if( sb1 == sizeof(npy_float64) && sb2 == sizeof(npy_int32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = pow(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = pow(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
sin_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sin(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sin(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
sin_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sin(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sin(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
sinh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sinh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sinh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
sinh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sinh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sinh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
sqrt_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sqrt(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sqrt(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
sqrt_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = sqrt(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = sqrt(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
tan_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
tan_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
tanh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tanh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tanh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
tanh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tanh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tanh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fpclassify_if( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fpclassify(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fpclassify(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fpclassify_id( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fpclassify(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fpclassify(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isfinite_1f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isfinite(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isfinite(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isfinite_1d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isfinite(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isfinite(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isinf_1f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isinf(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isinf(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isinf_1d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isinf(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isinf(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isnan_1f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isnan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isnan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isnan_1d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isnan(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isnan(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isnormal_1f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isnormal(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isnormal(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
isnormal_1d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = isnormal(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = isnormal(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
signbit_1f( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = signbit(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = signbit(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
signbit_1d( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_bool *dest = (npy_bool *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = signbit(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = signbit(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arccosh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = acosh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = acosh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arccosh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = acosh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = acosh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arcsinh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = asinh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = asinh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arcsinh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = asinh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = asinh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arctanh_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atanh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atanh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
arctanh_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = atanh(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = atanh(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cbrt_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cbrt(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cbrt(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
cbrt_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = cbrt(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = cbrt(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
copysign_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = copysign(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = copysign(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
copysign_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = copysign(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = copysign(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
erf_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = erf(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = erf(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
erf_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = erf(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = erf(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
erfc_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = erfc(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = erfc(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
erfc_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = erfc(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = erfc(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
exp2_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = exp2(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = exp2(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
exp2_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = exp2(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = exp2(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
expm1_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = expm1(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = expm1(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
expm1_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = expm1(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = expm1(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
fdim_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fdim(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fdim(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fdim_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fdim(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fdim(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fma_ffff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fma(x1[J], x2[J], x3[J]); 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    sb3 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fma(x1[J*sb1], x2[J*sb2], x3[J*sb3]); 
    }
    }
    return 0;
    }


static int
fma_dddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fma(x1[J], x2[J], x3[J]); 
}
        return 0;
    } else { // Strided
        sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    sb3 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fma(x1[J*sb1], x2[J*sb2], x3[J*sb3]); 
    }
    }
    return 0;
    }


static int
fmax_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmax(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmax(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fmax_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmax(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmax(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fmin_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmin(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmin(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
fmin_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = fmin(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = fmin(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
hypot_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = hypot(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = hypot(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
hypot_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = hypot(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = hypot(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
ilogb_if( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = ilogb(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = ilogb(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
ilogb_id( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = ilogb(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = ilogb(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lgamma_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lgamma(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lgamma(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lgamma_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lgamma(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lgamma(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log1p_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log1p(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log1p(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log1p_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log1p(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log1p(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log2_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log2(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log2(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
log2_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = log2(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = log2(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
logb_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = logb(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = logb(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
logb_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = logb(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = logb(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lrint_lf( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lrint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lrint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lrint_ld( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lrint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lrint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lround_lf( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lround(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lround(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
lround_ld( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = lround(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = lround(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
nearbyint_lf( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nearbyint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nearbyint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
nearbyint_ld( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int64 *dest = (npy_int64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nearbyint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nearbyint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
nextafter_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nextafter(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nextafter(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
nextafter_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nextafter(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nextafter(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
nexttoward_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nexttoward(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nexttoward(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
nexttoward_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = nexttoward(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = nexttoward(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
remainder_fff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = remainder(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = remainder(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
remainder_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = remainder(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = remainder(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
rint_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = rint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = rint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
rint_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = rint(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = rint(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
round_if( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = round(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = round(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
round_id( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_int32 *dest = (npy_int32 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = round(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = round(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
scalbln_ffl( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = scalbln(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    sb2 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = scalbln(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
scalbln_ddl( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = scalbln(x1[J], x2[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    sb2 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = scalbln(x1[J*sb1], x2[J*sb2]); 
    }
    }
    return 0;
    }


static int
tgamma_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tgamma(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tgamma(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
tgamma_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = tgamma(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = tgamma(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
trunc_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float32 *dest = (npy_float32 *)params->registers[store_in].mem;
    npy_float32 *x1 = (npy_float32 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float32) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = trunc(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float32);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = trunc(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
trunc_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{   
    NE_REGISTER store_in = params->program[pc].ret;
    NE_REGISTER arg1 = params->program[pc].arg1;
    BOUNDS_CHECK(store_in);
    BOUNDS_CHECK(arg1);

    npy_float64 *dest = (npy_float64 *)params->registers[store_in].mem;
    npy_float64 *x1 = (npy_float64 *)params->registers[arg1].mem;
    npy_intp sb1 = params->registers[arg1].stride;
        
    if( sb1 == sizeof(npy_float64) ) { // Aligned
        for(npy_intp J = 0; J < blocksize; J++) { 
    dest[J] = trunc(x1[J]); 
}
        return 0;
    } else { // Strided
       sb1 /= sizeof(npy_float64);
    for(npy_intp J = 0; J < blocksize; J++) { 
        dest[J] = trunc(x1[J*sb1]); 
    }
    }
    return 0;
    }


static int
multest_ddd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        NE_REGISTER arg2 = params->program[pc].arg2;
        
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);
        BOUNDS_CHECK(arg2);
        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        npy_intp sb1 = params->registers[arg1].stride;
        char *x2 = params->registers[arg2].mem;
        npy_intp sb2 = params->registers[arg2].stride;
        
        ne_mul(blocksize, (npy_float64 *)x1, (npy_float64 *)x2, (npy_float64 *)dest, sb1, sb2);
        return 0;
    }


static int
abs_fF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs(blocksize, (npy_complex64 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
abs_dD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_abs(blocksize, (npy_complex128 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
add_FFF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_add(blocksize, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
add_DDD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_add(blocksize, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
sub_FFF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_sub(blocksize, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
sub_DDD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_sub(blocksize, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
mult_FFF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_mul(blocksize, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
mult_DDD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_mul(blocksize, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
div_FFF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_div(blocksize, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
div_DDD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_div(blocksize, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
neg_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
neg_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_neg(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
conj_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_conj(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
conj_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_conj(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
conj_ff( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        fconj(blocksize, (npy_float32 *)x1, (npy_float32 *)dest);
        return 0;
    }


static int
conj_dd( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        fconj(blocksize, (npy_float64 *)x1, (npy_float64 *)dest);
        return 0;
    }


static int
sqrt_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sqrt(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sqrt_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sqrt(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log1p_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log1p(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log1p_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log1p(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
log10_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log10(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
log10_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_log10(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
exp_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_exp(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
exp_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_exp(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
expm1_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_expm1(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
expm1_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_expm1(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
pow_FFF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_pow(blocksize, (npy_complex64 *)x1, (npy_complex64 *)x2, (npy_complex64 *)dest);
        return 0;
    }


static int
pow_DDD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
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
        nc_pow(blocksize, (npy_complex128 *)x1, (npy_complex128 *)x2, (npy_complex128 *)dest);
        return 0;
    }


static int
arccos_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acos(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arccos_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acos(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arccosh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acosh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arccosh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_acosh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arcsin_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asin(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arcsin_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asin(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arcsinh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asinh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arcsinh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_asinh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arctan_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atan(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arctan_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atan(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
arctanh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atanh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
arctanh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_atanh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
cos_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cos(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
cos_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cos(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
cosh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cosh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
cosh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_cosh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
sin_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sin(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sin_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sin(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
sinh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sinh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
sinh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_sinh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
tan_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tan(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
tan_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tan(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


static int
tanh_FF( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tanh(blocksize, (npy_complex64 *)x1, (npy_complex64 *)dest);
        return 0;
    }


static int
tanh_DD( npy_intp blocksize, npy_intp pc, const NumExprObject *params )
{
        NE_REGISTER store_in = params->program[pc].ret;
        NE_REGISTER arg1 = params->program[pc].arg1;
        BOUNDS_CHECK(store_in);
        BOUNDS_CHECK(arg1);

        char *dest = params->registers[store_in].mem;
        char *x1 = params->registers[arg1].mem;
        nc_tanh(blocksize, (npy_complex128 *)x1, (npy_complex128 *)dest);
        return 0;
    }


