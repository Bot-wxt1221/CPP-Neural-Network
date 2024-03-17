// #pragma GCC target("avx")
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-fwhole-program")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-funsafe-loop-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
#include<bits/stdc++.h>
// #define ONLINE_JUDGE
#define INPUT_DATA_TYPE int
#define OUTPUT_DATA_TYPE int
inline INPUT_DATA_TYPE read(){register INPUT_DATA_TYPE x=0;register char f=0,c=getchar();while(c<'0'||'9'<c)f=(c=='-'),c=getchar();while('0'<=c&&c<='9')x=(x<<3)+(x<<1)+(c&15),c=getchar();return f?-x:x;}void print(OUTPUT_DATA_TYPE x){if(x<0)x=-x,putchar('-');if(x>9)print(x/10);putchar(x%10^48);return;}


/*
INPUT_SIZE:  输入向量长度
OUTPUT_SIZE: 输出向量长度
NET_DEEP:    中间层深度
NET_SIZE:    中间层向量长度
*/
const int INPUT_SIZE=784,OUTPUT_SIZE=10,NET_DEEP=10,NET_SIZE=800,MAX_PIC=1000010;
std::mt19937 mt(time(0));

inline double sigmoid(double x,char flg) {
    if(flg) return sigmoid(x,0)*(1-sigmoid(x,0));
	return (double)(1.0)/(1+std::exp(-x));
}

inline double ReLU(double x,char flg){
    if(flg) return x<0?0:1;
    return x<0?0:x;
}

const double B_ELU=1;
inline double ELU(double x,char flg){
    if(flg) return x<0?(std::exp(x)*B_ELU):1;
    return x<0?((std::exp(x)-1)*B_ELU):x;
}

const double B_LeakyReLU=0.01;
inline double LeakyReLU(double x,char flg){
    if(flg) return x<0?B_LeakyReLU:1;
    return x<0?x*B_LeakyReLU:x;
}

inline double square(double x){return x*x;}

inline double logistic(double x){return LeakyReLU(x,0);}
inline double dlogistic(double x){return LeakyReLU(x,1);}

#define MATRIX_DATA_TYPE double
struct MATRIX{
    int n,m;
    std::vector<std::vector<double>> mat;

    /*
    flg=0  set0
    flg=1 set random [a,b]
    flg=2 norm 
    
    */
    inline void init(int _n,int _m,char flg,double a=0,double b=0){
        n=_n,m=_m;
        register int i,j;
        std::uniform_real_distribution<double> rd1(a,b);
        std::normal_distribution<double> rd2(a,b);
        
        mat.clear();
        mat.resize(n);
        for(i=0;i<n;++i){
            mat[i].resize(m,0);
            if(flg==1) for(j=0;j<m;++j) mat[i][j]=rd1(mt);
            else if(flg==2) for(j=0;j<m;++j) mat[i][j]=rd2(mt);
        }
        mat.shrink_to_fit();
    }

    inline MATRIX operator + (const MATRIX &a) const {
        register int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        assert(n==a.n&&m==a.m);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]+a.mat[i][j];

        return ans;
    }

    inline MATRIX operator * (const MATRIX &a) const {
        register int i,j,k;
        MATRIX ans;
        ans.init(n,a.m,0);
        if(m!=a.n) assert(0);
        for(i=0;i<n;++i)
            for(k=0;k<m;++k)
                #pragma unroll(32)
                for(j=0;j<a.m;++j)
                    ans.mat[i][j]+=mat[i][k]*a.mat[k][j];
        return ans;
    }

    inline MATRIX operator * (const double &a) const {
        register int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]*a;
        return ans;
    }

    inline MATRIX operator % (const MATRIX &a) const {
        register int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        assert(n==a.n&&m==a.m);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=mat[i][j]*a.mat[i][j];
        return ans;
    }

    inline MATRIX T() const{
        register int i,j;
        MATRIX ans;
        ans.init(m,n,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[j][i]=mat[i][j];
        return ans;
    }

    inline MATRIX logis() const{
        register int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=logistic(mat[i][j]);
        return ans;
    }

    inline MATRIX dlogis() const{
        register int i,j;
        MATRIX ans;
        ans.init(n,m,0);
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                ans.mat[i][j]=dlogistic(mat[i][j]);
        return ans;
    }

    inline void print() const{
        register int i,j;
        for(i=0;i<n;++i,putchar('\n'))
            for(j=0;j<m;++j,putchar(' '))
                printf("%.20lf",mat[i][j]);
    }

    inline void getMat(){
        register int i,j;
        for(i=0;i<n;++i)
            for(j=0;j<m;++j)
                scanf("%lf",&mat[i][j]);
    }
}B[NET_DEEP+10],W[NET_DEEP+10],Z[NET_DEEP+10],ZL[NET_DEEP+10],DB[NET_DEEP+10],DW[NET_DEEP+10],DZ[NET_DEEP+10],DZL[NET_DEEP+10],GRP_DW[NET_DEEP+10],GRP_DB[NET_DEEP+10],inputPic[MAX_PIC];

inline MATRIX work(MATRIX input){
    register int i;
    Z[1]=input*W[0];
    for(i=1;i<=NET_DEEP;++i){
        Z[i]=Z[i]+B[i];
        ZL[i]=Z[i].logis();
        Z[i+1]=ZL[i]*W[i];
    }
    return Z[NET_DEEP+1];
}

inline double getCost(MATRIX output,MATRIX ans){
    register int i;
    register double res=0;
    for(i=0;i<OUTPUT_SIZE;++i) res+=square(ans.mat[0][i]-output.mat[0][i]);
    return res;
}

inline void getDelta(MATRIX ans){
    register int l,i,j;
    for(i=0;i<OUTPUT_SIZE;++i) DZ[NET_DEEP+1].mat[0][i]=-2*(Z[NET_DEEP+1].mat[0][i]-ans.mat[0][i]);
    for(l=NET_DEEP;l;--l){
        for(i=0;i<W[l].n;++i)
            for(j=0;j<W[l].m;++j)
                DW[l].mat[i][j]=W[l].mat[i][j]*DZ[l+1].mat[0][j];
        DZL[l]=DZ[l+1]*(W[l].T());
        DZ[l]=Z[l].dlogis()%DZL[l];
        DB[l]=DZ[l];
    }
    for(i=0;i<W[0].n;++i)
        for(j=0;j<W[0].m;++j)
            DW[0].mat[i][j]=W[0].mat[i][j]*DZ[1].mat[0][j];
    return;
}

inline void addDelta(double w){
    register int i;
    W[0]=W[0]+(GRP_DW[0]*(w));
    for(i=1;i<=NET_DEEP;++i){
        B[i]=B[i]+(GRP_DB[i]*(w));
        W[i]=W[i]+(GRP_DW[i]*(w));
    }

    GRP_DW[0].init(INPUT_SIZE,NET_SIZE,0);
    for(i=1;i<NET_DEEP;++i){
        GRP_DB[i].init(1,NET_SIZE,0);
        GRP_DW[i].init(NET_SIZE,NET_SIZE,0);
    }
    GRP_DB[NET_DEEP].init(1,NET_SIZE,0);
    GRP_DW[NET_DEEP].init(NET_SIZE,OUTPUT_SIZE,0);
    return;
}

inline void addGroupDelta(){
    register int i;
    GRP_DW[0]=GRP_DW[0]+DW[0];
    for(i=1;i<=NET_DEEP;++i){
        GRP_DB[i]=GRP_DB[i]+DB[i];
        GRP_DW[i]=GRP_DW[i]+DW[i];
    }
    return;
}

inline void init(){
    register int i;
    register double BrdA=0,BrdB=0.1;register int flgB=1;
    register double WrdA=0,WrdB=std::sqrt(2.0/INPUT_SIZE);register int flgW=2;
    
    W[0].init(INPUT_SIZE,NET_SIZE,flgW,WrdA,WrdB);
    DW[0]=W[0];
    Z[0].init(1,INPUT_SIZE,0);
    
    for(i=1;i<NET_DEEP;++i){
        Z[i].init(1,NET_SIZE,flgB,BrdA,BrdB);
        B[i]=ZL[i]=DB[i]=DZL[i]=Z[i]=Z[i];
        W[i].init(NET_SIZE,NET_SIZE,flgW,WrdA,WrdB);
        DW[i]=W[i];
    }
    Z[NET_DEEP].init(1,NET_SIZE,flgB,BrdA,BrdB);
    B[NET_DEEP]=ZL[NET_DEEP]=DB[NET_DEEP]=DZL[NET_DEEP]=Z[NET_DEEP]=Z[NET_DEEP];
    W[NET_DEEP].init(NET_SIZE,OUTPUT_SIZE,flgW,WrdA,WrdB);
    DW[NET_DEEP]=W[NET_DEEP];

    Z[NET_DEEP+1].init(1,OUTPUT_SIZE,0);
    DZ[NET_DEEP+1]=Z[NET_DEEP+1];

    GRP_DW[0].init(INPUT_SIZE,NET_SIZE,0);
    for(i=1;i<NET_DEEP;++i){
        GRP_DB[i].init(1,NET_SIZE,0);
        GRP_DW[i].init(NET_SIZE,NET_SIZE,0);
    }
    GRP_DB[NET_DEEP].init(1,NET_SIZE,0);
    GRP_DW[NET_DEEP].init(NET_SIZE,OUTPUT_SIZE,0);
    return;
}

inline void save(){
    freopen("result", "w", stdout);
    register int i;
    W[0].print();
    for(i=1;i<=NET_DEEP;++i)
        W[i].print();
    for(i=1;i<=NET_DEEP;++i)
        B[i].print();
    return;
}
inline void getNet(){
    freopen("result", "r", stdin);
    register int i;
    W[0].getMat();
    for(i=1;i<=NET_DEEP;++i) W[i].getMat();
    for(i=1;i<=NET_DEEP;++i) B[i].getMat();
    return;
}


/*
first:input matrix
second:ans matrix
*/
// inline std::pair<MATRIX,MATRIX> getData(){
//     register int i,a=mt()&1;
//     // a=1;
//     MATRIX input,ans;
//     input.init(1,INPUT_SIZE,0);
//     ans.init(1,OUTPUT_SIZE,0);
    
//     for(i=0;i<INPUT_SIZE;++i) input.mat[0][i]=i^a;
//     for(i=0;i<OUTPUT_SIZE;++i) ans.mat[0][i]=input.mat[0][(int)std::min(INPUT_SIZE-1.0,1.0*i*INPUT_SIZE/OUTPUT_SIZE)];
    
//     return {input,ans};
// }
int res[100010],now=0,cntPic;
void getPic(int n){
    cntPic=n,now=0;
    register int i=0,j;
    freopen("lables", "r", stdin);
    for(i=0;i<cntPic;++i) res[i]=read();
    freopen("images", "r", stdin);
    for(i=0;i<cntPic;++i){
        inputPic[i].init(1,INPUT_SIZE,0);
        for(j=0;j<INPUT_SIZE;++j) inputPic[i].mat[0][j]=read()*1.0/255;
    }
}
inline std::pair<MATRIX,MATRIX> getData(char rd){
    if(rd){
        std::uniform_int_distribution<int> rd(0,cntPic-1);
        now=rd(mt);
    }else ++now;
    now%=cntPic;
    MATRIX input,ans;
    input.init(1,INPUT_SIZE,0);
    ans.init(1,OUTPUT_SIZE,0);
    
    input=inputPic[now];
    ans.mat[0][res[now]]=1;
    return {input,ans};
}
int getOutputAns(MATRIX output){
    register int i,res=0;
    for(i=0;i<10;++i)
        if(output.mat[0][i]>output.mat[0][res]) res=i;
    return res;
}

void train(int cnt,int grp,int outGap,double learn,char randomInput){
    register int i,cntRight=0,cntAll=0;
    double time;
    clock_t start=clock(),nowtime;
    MATRIX input,output,ans;

    for(i=0;i<=cnt;++i){
        if(i%1000==0) cntRight=cntAll=0;
        auto pr=getData(randomInput);
        input=pr.first,ans=pr.second;
        output=work(input);

        cntRight+=(getOutputAns(output)==res[now]);
        ++cntAll;
        
        if(i%outGap==0){
            ans.print();
            output.print();
            
            nowtime=clock()-start;
            time=nowtime*1.0/CLOCKS_PER_SEC/60;
            printf("train case%d %lf %lf\nuse time:%.3lfmins all time:%.3lfmins left time:%.3lfmins\n",i,getCost(output,ans),cntRight*1.0/cntAll,time,time/(i+1)*cnt,time/(i+1)*cnt-time);
        }

        getDelta(ans);
        addGroupDelta();
        if(i%grp==0) addDelta(learn/grp);
        
    }
}


void getTestPic(int n){
    cntPic=n,now=0;
    register int i=0,j;
    freopen("test_lables", "r", stdin);
    for(i=0;i<cntPic;++i) res[i]=read();
    freopen("test_images", "r", stdin);
    for(i=0;i<cntPic;++i){
        inputPic[i].init(1,INPUT_SIZE,0);
        for(j=0;j<INPUT_SIZE;++j) inputPic[i].mat[0][j]=read()*1.0/255;
    }
}

void test(int n,int all,int outGap){
    register int i=0,cntRight=0,cntAll=0;
    clock_t start=clock(),nowtime;
    double time;
    MATRIX input,output,ans;
    getTestPic(all);
    for(i=0;i<n;++i){
        auto pr=getData(1);
        input=pr.first,ans=pr.second;
        output=work(input);

        cntRight+=(getOutputAns(output)==res[now]);
        ++cntAll;
        if(i%10==0){
            nowtime=clock()-start;
            time=nowtime*1.0/CLOCKS_PER_SEC/60;
            printf("test case%d %lf %lf\nuse time:%.3lfmins all time:%.3lfmins left time:%.3lfmins\n",i,getCost(output,ans),cntRight*1.0/cntAll,time,time/(i+1)*n,time/(i+1)*n-time);
        }
    }
    return;
}

int main(){
	#ifndef ONLINE_JUDGE
	freopen("name.in", "r", stdin);
	freopen("name.out", "w", stdout);
	#endif

    register int i;
    MATRIX input,output,ans;
    init();
    getPic(60000);
    getNet();
    train(60000,5,10,0.001,1);
    test(5000,10000,50);
    save();

	#ifndef ONLINE_JUDGE
	fclose(stdin);
	fclose(stdout);
	#endif
    return 0;
}