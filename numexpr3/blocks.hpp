/*********************************************************************
  Numexpr - Fast numerical array expression evaluator for NumPy.

      License: MIT
      Author:  See AUTHORS.txt

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

#ifdef USE_VML
// The values below have been tuned for a nowadays Core2 processor 
// Note: with VML functions a larger block size (e.g. 4096) allows to make use
// of the automatic multithreading capabilities of the VML library 
#define BLOCK_SIZE1 4096
#define BLOCK_SIZE2 32
#else
// The values below have been tuned for a nowadays Core2 processor
// Note: without VML available a smaller block size is best, specially
// for the strided and unaligned cases.  Recent implementation of
// multithreading make it clear that larger block sizes benefit
// performance (although it seems like we don't need very large sizes
// like VML yet). 
#define BLOCK_SIZE1 1024
#define BLOCK_SIZE2 16
#endif
