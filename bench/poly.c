/* ####################################################################### */
/* This script compares the speed of the computation of a polynomial       */
/* in C in a couple of different ways.                                     */
/*                                                                         */
/* Author: Francesc Alted                                                  */
/* Date: 2010-02-05                                                        */
/* ####################################################################### */


#include <stdio.h>
#include <math.h>
#if defined(_WIN32) && !defined(__MINGW32__)
  #include <time.h>
  #include <windows.h>
#else
  #include <unistd.h>
  #include <sys/time.h>
#endif


#define N  10*1000*1000

double x[N];
double y[N];


#if defined(_WIN32) && !defined(__MINGW32__)

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;

  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);

    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;

    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS;
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }

  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }

  return 0;
}
#endif   /* _WIN32 */


/* Given two timeval stamps, return the difference in seconds */
float getseconds(struct timeval last, struct timeval current) {
  int sec, usec;

  sec = current.tv_sec - last.tv_sec;
  usec = current.tv_usec - last.tv_usec;
  return (float)(((double)sec + usec*1e-6));
}

int main(void) {
  int i;
  double inf = -1;
  struct timeval last, current;
  float tspend;

  for(i=0; i<N; i++) {
    x[i] = inf+(2.*i)/N;
  }

  gettimeofday(&last, NULL);
  for(i=0; i<N; i++) {
    //y[i] = .25*pow(x[i],3.) + .75*pow(x[i],2.) - 1.5*x[i] - 2;
    y[i] = ((.25*x[i] + .75)*x[i] - 1.5)*x[i] - 2;
  }
  gettimeofday(&current, NULL);
  tspend = getseconds(last, current);
  printf("Compute time:\t %.3fs\n", tspend);

}
