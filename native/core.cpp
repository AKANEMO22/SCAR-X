// Original SCAR-X CPU kernel. Same equations as neural/core.py, no autograd.
#include <algorithm>
#include <cmath>
#include <vector>
#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

static float dot(const float* a,const float* b,int n) {
    float sum=0; int i=0;
#ifdef __AVX2__
    __m256 v=_mm256_setzero_ps();
    for(;i+8<=n;i+=8) v=_mm256_add_ps(v,_mm256_mul_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));
    float t[8]; _mm256_storeu_ps(t,v); for(float x:t) sum+=x;
#endif
    for(;i<n;i++) sum+=a[i]*b[i];
    return sum;
}
static float clip(float x,float c) { return std::clamp(x,-c,c); }
// a: win,wrec,wout,bias,stablein,stablerec,stableout,stablebias,
// voltage,adaptation,spikes,pre,post,features,elig_in,elig_rec.
static void probs(int h,int o,float** a,float* out) {
    float high=-INFINITY;
    for(int j=0;j<o;j++) { out[j]=dot(a[2]+j*h,a[13],h)+a[3][j]; high=std::max(high,out[j]); }
    float sum=0; for(int j=0;j<o;j++) { out[j]=std::exp(out[j]-high); sum+=out[j]; }
    for(int j=0;j<o;j++) out[j]/=sum;
}
API void scar_step(int d,int h,int o,int k,int r,const float* p,float** a,const int* idx,const int* rows,const float* x,int track,int rate,float* out) {
    std::vector<float> old(a[10],a[10]+h);
    for(int j=0;j<h;j++) {
        float current=dot(a[0]+j*d,x,d);
        for(int q=0;q<k;q++) current+=a[1][j*k+q]*old[idx[j*k+q]];
        a[8][j]=p[0]*a[8][j]+current;
        float limit=p[1]+p[3]*a[9][j];
        a[10][j]=rate?std::tanh(a[8][j]):(a[8][j]>=limit?1.f:0.f);
        if(!rate) a[8][j]-=a[10][j]*limit;
        a[9][j]=p[2]*a[9][j]+a[10][j];
        a[13][j]=p[5]*a[13][j]+(1-p[5])*a[10][j];
    }
    if(track) {
        for(int rr=0;rr<r;rr++) {
            int j=rows[rr];
            for(int q=0;q<d;q++) a[14][rr*d+q]=clip(p[4]*a[14][rr*d+q]+a[10][j]*a[11][q]-.5f*a[12][j]*x[q],5.f);
            for(int q=0;q<k;q++) { int pre=idx[j*k+q]; a[15][rr*k+q]=clip(p[4]*a[15][rr*k+q]+a[10][j]*a[11][d+pre]-.5f*a[12][j]*old[pre],5.f); }
        }
        for(int j=0;j<d;j++) a[11][j]=p[4]*a[11][j]+x[j];
        for(int j=0;j<h;j++) { a[11][d+j]=p[4]*a[11][d+j]+old[j]; a[12][j]=p[4]*a[12][j]+a[10][j]; }
    }
    probs(h,o,a,out);
}
API float scar_learn(int d,int h,int o,int k,int r,const float* p,float** a,const int* rows,int target,int mode,double* reward_mean,int has_reward,double reward) {
    std::vector<float> prob(o); probs(h,o,a,prob.data());
    float loss=-std::log(std::max(prob[target],1e-12f));
    if(mode==0) return loss;
    for(int j=0;j<o;j++) {
        float error=(j==target?1.f:0.f)-prob[j];
        for(int q=0;q<h;q++) { int z=j*h+q; a[2][z]+=clip(p[6]*error*a[13][q]-p[8]*(a[2][z]-a[6][z]),p[9]); a[2][z]=clip(a[2][z],p[10]); }
        a[3][j]+=clip(p[6]*error-p[8]*(a[3][j]-a[7][j]),p[9]); a[3][j]=clip(a[3][j],p[10]);
    }
    double raw=has_reward?reward:prob[target];
    float modulation=mode==2?1.f:float(std::clamp(raw-*reward_mean,-1.,1.));
    *reward_mean=.95*(*reward_mean)+.05*raw;
    if(mode==2 || mode==3) for(int rr=0;rr<r;rr++) {
        int j=rows[rr];
        for(int q=0;q<d;q++) { int z=j*d+q; a[0][z]+=clip(p[7]*modulation*a[14][rr*d+q]-p[8]*(a[0][z]-a[4][z]),p[9]); }
        for(int q=0;q<k;q++) { int z=j*k+q; a[1][z]+=clip(p[7]*modulation*a[15][rr*k+q]-p[8]*(a[1][z]-a[5][z]),p[9]); }
    }
    if(mode==2 || mode==3) { for(int j=0;j<h*d;j++) a[0][j]=clip(a[0][j],p[10]); for(int j=0;j<h*k;j++) a[1][j]=clip(a[1][j],p[10]); }
    return loss;
}
