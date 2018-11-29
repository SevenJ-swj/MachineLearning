///2018/11/28
///Auhtor:Wenjun Shi
///All Rights Reserved

///Realize SoftMargin SVM with kernels Through SMO Algorithm
#include <bits/stdc++.h>
#define Mat vector<vector<double> >

using namespace std;

Mat TrainDataSet,TrainLabelSet,TestDataSet,TestLabelSet;
const int BiThreshold=50;
const int ImageSize=28*28;

Mat XX;
vector<double> YY;

map<int,map<int,double> >K;
Mat transport(Mat A);
int Binarization(int pixel)
{
    if(pixel>BiThreshold) return 1;
    else return 0;
}

void normalization(Mat &X)
{
    for(int col=0;col<X[0].size();col++)
    {
        double sum=0,sumsqr=0;
        for(int row=0;row<X.size();row++)
        {
            sum+=X[row][col];
            sumsqr+=X[row][col];
        }
        double u=sum/X.size();
    }
}

void read_label_set(char *path,Mat &LabelSet)
{
    char tmp;
    freopen(path,"rb",stdin);
    for(int i=0; i<8; i++) scanf("%c",&tmp);
    unsigned char lab;
    while(~scanf("%c",&lab))
    {
        vector<double> t;
        t.push_back((double)lab);
        LabelSet.push_back(t);
    }
}
void printMat(Mat A)
{
    for(int i=0; i<A.size(); i++)
    {
        for(int j=0; j<A[i].size(); j++)
        {
            cout<<A[i][j]<<' ';
        }
        cout<<endl;
    }
    cout<<"row:"<<A.size()<<' '<<"col:"<<A[0].size()<<endl;
}
void printImage(vector<double> img)
{
    for(int i=0; i<28; i++)
    {
        for(int j=0; j<28; j++)
        {
            if(img[i*28+j]>0) cout<<"*";
            else cout<<"-";
        }
        cout<<endl;
    }
    cout<<endl;
}
void read_data_set(char* path,Mat &DataSet)
{
    vector<double> image;
    char tmp;
    freopen(path,"rb",stdin);
    for(int i=0; i<16; i++) scanf("%c",&tmp);
    unsigned char pix;
    while(~scanf("%c",&pix))
    {
        pix=(char)Binarization(pix);
        image.push_back((double)pix);
        if(image.size()==ImageSize)  DataSet.push_back(image),image.clear();
    }
    // DataSet = transport(DataSet);
}

void read()
{
    read_data_set("E:\\data\\train-images.idx3-ubyte",TrainDataSet);
    read_data_set("E:\\data\\t10k-images.idx3-ubyte",TestDataSet);
    read_label_set("E:\\data\\train-labels.idx1-ubyte",TrainLabelSet);
    read_label_set("E:\\data\\t10k-labels.idx1-ubyte",TestLabelSet);
    cout<<"Input result£º"<<endl;
    cout<<"Number of Train Data:"<<TrainDataSet.size()<<endl;
    cout<<"Number of Train Label:"<<TrainLabelSet.size()<<endl;
    cout<<"Number of Test Data:"<<TestDataSet.size()<<endl;
    cout<<"Number of Test Label:"<<TestLabelSet.size()<<endl;
    cout<<"Input data verify:"<<endl;
    cout<<"corresponding label:"<<TrainLabelSet[0][0]<<endl;
    printImage(TrainDataSet[0]);
    cout<<"corresponding label:"<<TestLabelSet[0][0]<<endl;
    printImage(TestDataSet[0]);
}

vector<double> operator * (vector<double> &a,double b)
{
    int n=a.size();
    vector<double> res(n,0);
    for(int i=0; i<n; i++)
        res[i]=a[i]*b;
    return res;
}
vector<double> operator - (vector<double> a,vector<double> b)
{
    int n=a.size();
    vector<double> res(n,0);
    for(int i=0; i<n; i++)
        res[i]=a[i]-b[i];
    return res;
}

inline Mat operator + (Mat a,Mat b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[0].size(); j++)
        {
            c[i][j]=a[i][j]+b[i][j];
        }
    }
    return c;
}

inline Mat operator - (Mat a,Mat b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[0].size(); j++)
        {
            c[i][j]=a[i][j]-b[i][j];
        }
    }
    return c;
}

inline Mat operator * (Mat a,double b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0; i<a.size(); i++)
    {
        for(int j=0; j<a[0].size(); j++)
        {
            c[i][j]=a[i][j]*b;
        }
    }
    return c;
}


Mat getEMat(int n,double lamda)
{
    Mat a(n,vector<double>(n,0));
    for(int i=0; i<n; i++)
    {
        a[i][i]=lamda;
    }
    return a;
}

Mat inverse(Mat A)
{
    int n=A.size();
    Mat C(n,vector<double>(n,0));
    for(int i=0; i<n; i++) C[i][i]=1;
    for(int i=0; i<n; i++)
    {
        for(int j=i; j<n; j++)
        {
            if(fabs(A[j][i])>0)
            {
                swap(A[i],A[j]);
                swap(C[i],C[j]);
                break;
            }
        }
        C[i]=C[i]*(1/A[i][i]);
        A[i]=A[i]*(1/A[i][i]);
        for(int j=0; j<n; j++)
        {
            if(j!=i&&fabs(A[j][i])>0)
            {
                C[j]=C[j]-C[i]*A[j][i];
                A[j]=A[j]-A[i]*A[j][i];
            }
        }
    }
    return C;
}

Mat transport(Mat A)
{
    Mat C(A[0].size(),vector<double>(A.size(),0));
    for(int i=0; i<A.size(); i++)
    {
        for(int j=0; j<A[0].size(); j++)
        {
            C[j][i]=A[i][j];
        }
    }
    return C;
}

Mat dot(Mat A,Mat B)
{
    Mat C(A.size(),vector<double>(B[0].size(),0));
    for(int i=0; i<A.size(); i++)
    {
        for(int j=0; j<B[0].size(); j++)
        {
            for(int k=0; k<A[i].size(); k++)
            {
                C[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
    return C;
}


double eps=1e-5;

double kernel(vector<double> &x1,vector<double> &x2)
{
    double sum=0;
    for(int i=0; i<x1.size(); i++)
         //sum+=(x1[i]-x2[i])*(x1[i]-x2[i]);
        sum+=x1[i]*x2[i];
    //cout<<"kernel"<<sum<<endl;
    return sum;
   // return exp(-1.3*sum);
}


double G(vector<double> &x1,vector<double> &y,double *alpha,double b,Mat &x)
{
    double sum=0;
    for(int i=0; i<x.size(); i++)
    {
        //  cout<<i<<endl;
        sum+=alpha[i]*y[i]*kernel(x1,x[i]);
       // sum+=alpha[i]*y[i]*kernel(x1,x[i]);
    }
    return sum+b;
}

int takestep(double *E,int aSize,vector<double> &y,double *alpha,double &b,int a1_id,int a2_id,double vC,Mat &x)
{
    double L,H;
    if(y[a1_id]!=y[a2_id]) L=max(0.0,alpha[a2_id]-alpha[a1_id]),H=min(vC,vC+alpha[a2_id]-alpha[a1_id]);
    else L=max(0.0,alpha[a2_id]+alpha[a1_id]-vC),H=min(vC,alpha[a2_id]+alpha[a1_id]);

    // cout<<"fenzi:"<<(y[a2_id]*(E[a1_id]-E[a2_id]))<<endl;
    // cout<<"fenmu:"<<kernel(x[a1_id],x[a1_id])<<' '<<kernel(x[a2_id],x[a2_id])<<' '<<-2*kernel(x[a1_id],x[a2_id])<<endl;
    double yita=K[a1_id][a1_id]+K[a2_id][a2_id]-2*K[a1_id][a2_id];
    //cout<<yita<<endl;
   // if(yita<=eps) return 0;
                        // kernel(x[a2_id],x[a2_id])-2*kernel(x[a1_id],x[a2_id]));
    if(yita<=eps) return 0;
    double newA2=alpha[a2_id]+
                 (y[a2_id]*(E[a1_id]-E[a2_id]))/yita;
   // cout<<y[a2_id]<<' '<<E[a1_id]<<' '<<E[a2_id]<<' '<<"ETA:"<<yita<<endl;
   //   cout<<"id:"<<a2_id<<" newA2:"<<newA2<<endl;
    //  cout<<"L,H:"<<L<<' '<<H<<endl;
    if(newA2<L) newA2=L;
    if(newA2>H) newA2=H;
    if(fabs(L-H)<eps) return 0;
    if(fabs(newA2-alpha[a2_id])<1e-8) return 0;
    double newA1=alpha[a1_id]+y[a1_id]*y[a2_id]*(alpha[a2_id]-newA2);
   //   cout<<"a1_id:"<<a1_id<<" newA1:"<<newA1<<endl;
    //  cout<<"a2_id:"<<a2_id<<' '<<"newA2:"<<newA2<<endl;

   /* double newb1=0,newb2=0;
    if(newA1>eps&&newA1<vC-eps) b=-E[a1_id]-y[a1_id]*K[a1_id][a1_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a1_id]*(newA2-alpha[a2_id])+b;
    else if(newA2>eps&&newA2<vC-eps) b=-E[a2_id]-y[a1_id]*K[a1_id][a2_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a2_id]*(newA2-alpha[a2_id])+b;
    newb1=-E[a1_id]-y[a1_id]*K[a1_id][a1_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a1_id]*(newA2-alpha[a2_id])+b;
    newb2=-E[a2_id]-y[a1_id]*K[a1_id][a2_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a2_id]*(newA2-alpha[a2_id])+b;
    cout<<"newb1:"<<' '<<newb1<<" newb2:"<<' '<<newb2<<endl;*/
  /* for(int i=0;i<aSize;i++){
        if(a1_id==i||a2_id==i) continue;
        newb1-=alpha[i]*y[i]*K[i][a1_id];
        newb2-=alpha[i]*y[i]*K[i][a2_id];
   }*/
  // newb1+=y[a1_id]-newA1*y[a1_id]*K[a1_id][a1_id]-newA2*y[a2_id]*K[a2_id][a1_id];
  // newb2+=y[a2_id]-newA1*y[a1_id]*K[a1_id][a1_id]-newA2*y[a2_id]*K[a2_id][a1_id];


   double maxF=-1e9,minF=1e9;
    for(int i=0;i<aSize;i++)
    {
        double vy=0;
        for(int j=0; j<aSize; j++)
        {
         if(j==a1_id) vy+=newA1*y[j]*K[i][j];
         else if(j==a2_id) vy+=newA2*y[j]*K[i][j];
         else vy+=alpha[j]*y[j]*K[i][j];
       // sum+=alpha[i]*y[i]*kernel(x1,x[i]);
        }
        if(y[i]==-1)
        {
            if(vy>maxF) maxF=vy;
        }
        else
            if(vy<minF) minF=vy;
    }
   // cout<<"MAX"<<maxF<<' '<<minF<<endl;
   // if(maxF<minF) b=0,cout<<"RRR"<<endl;
    b=-0.5*(maxF+minF);

    double newb1=0,newb2=0;
    if(newA1>eps&&newA1<vC-eps) {
        b=-E[a1_id]-y[a1_id]*K[a1_id][a1_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a1_id]*(newA2-alpha[a2_id])+b;
        //cout<<a1_id<<' '<<"A1b:"<<-E[a1_id]<<' '<<-y[a1_id]*K[a1_id][a1_id]*(newA1-alpha[a1_id])<<' '<<-y[a2_id]*K[a2_id][a1_id]*(newA2-alpha[a2_id])<<endl;
    }
    else if(newA2>eps&&newA2<vC-eps){
        b=-E[a2_id]-y[a1_id]*K[a1_id][a2_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a2_id]*(newA2-alpha[a2_id])+b;
        //cout<<"A2b:"<<-E[a2_id]<<' '<<-y[a1_id]*K[a1_id][a2_id]*(newA1-alpha[a1_id])<<' '<<-y[a2_id]*K[a2_id][a2_id]*(newA2-alpha[a2_id])<<endl;
    }
    else{

        newb1=-E[a1_id]-y[a1_id]*K[a1_id][a1_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a1_id]*(newA2-alpha[a2_id])+b;
        newb2=-E[a2_id]-y[a1_id]*K[a1_id][a2_id]*(newA1-alpha[a1_id])-y[a2_id]*K[a2_id][a2_id]*(newA2-alpha[a2_id])+b;
    b=(newb1+newb2)/2;
    }
    //cout<<"secB:"<<b<<' '<<G(x[a1_id],y,alpha,b,x)<<' '<<G(x[a2_id],y,alpha,b,x)<<endl;
    //
   // cout<<"firB:"<<-0.5*(maxF+minF)<<' '<<G(x[a1_id],y,alpha,b,x)<<' '<<G(x[a2_id],y,alpha,b,x)<<endl;

    // b=-0.5*(maxF+minF);

    alpha[a2_id]=newA2;
    alpha[a1_id]=newA1;
    for(int i=0; i<aSize; i++)
    {
        E[i]=G(x[i],y,alpha,b,x)-y[i];
    }
    return 1;
}

int upalpha(double *E,int aSize,vector<double> &y,double *alpha,double &b,int a1_id,double vC,Mat &x)
{

    //printf("E[i]:%f\n",E[a1_id]);
    int a2_id=-1;
    double maxEv=-1;
    for(int i=0; i<aSize; i++)
    {
        if(i==a1_id) continue;
        if(fabs(E[i]-E[a1_id])>maxEv)
        {
            a2_id=i;
            maxEv=fabs(E[i]-E[a1_id]);
        }
    }
    int f=0;
    f=takestep(E,aSize,y,alpha,b,a1_id,a2_id,vC,x);
    if(f==1) return f;
    a2_id=rand()%aSize;
    while(a2_id==a1_id) a2_id=rand()%aSize;
    f=takestep(E,aSize,y,alpha,b,a1_id,a2_id,vC,x);
    return f;
   // a2_id=rand()%aSize;
   // while(a2_id==a1_id) a2_id=rand()%aSize;
//cout<<"IJ:"<<a1_id<<' '<<a2_id<<endl;

    //for(int i=0; i<aSize; i++) cout<<alpha[i]<<' ';cout<<b<<endl;cout<<endl;
}

void SMO(Mat x,vector<double> y,double vC,double *alpha,double &b)
{
    int aSize=x.size();
    double E[aSize];
    cout<<"test"<<endl;
    memset(alpha,0,aSize*sizeof(double));
    cout<<"test2"<<endl;
    int a1_id=-1;
    for(int i=0; i<aSize; i++)
    {
        E[i]=G(x[i],y,alpha,b,x)-y[i];
    }
    cout<<endl;
    cout<<"test3"<<endl;
    //alpha[4]=alpha[16]=0.139509;
    //alpha[7]=alpha[22]=0.0159439;
    //b=5.24235;
    int t=0;
    int examineAll=1,change=0;
    while(t++<50000 && (change>0||examineAll==1))
    {
        if(t%1000==0) cout<<t<<endl;
        a1_id=-1;
        int change=0;
        if(examineAll==0){
        for(int i=0; i<aSize; i++)
        {
            if(alpha[i]>eps&&alpha[i]<vC-eps )
            {
                //cout<<"not Bound:"<<' '<<y[i]*G(x[i],y,alpha,b,x)<<' '<<i<<endl;
                a1_id=i;
                if (((y[i] * E[i] < -eps)&&(alpha[i] < vC-eps))||((y[i] * E[i] > eps) && (alpha[i] > eps)))
                    change+=upalpha(E,aSize,y,alpha,b,a1_id,vC,x);
            }
        }
        }
        else
        {

            for(int i=1; i<aSize; i++)
            {
                a1_id=i;
                if (((y[i] * E[i] < -eps)&&(alpha[i] < vC-eps))||((y[i] * E[i] > eps) && (alpha[i] > eps)))
                    change+=upalpha(E,aSize,y,alpha,b,a1_id,vC,x);
            }
        }
        if(examineAll==1)
        {
            examineAll=0;
        }
        else if(change==0)
        {
            examineAll=1;
        }
    }
    for(int i=0; i<aSize; i++) cout<<alpha[i]<<' ';
    cout<<b<<endl;
    cout<<endl;

}

int main2()
{

    double t1,t2,t3;
    ifstream fin2("E://traindata4.txt");
    while(fin2>>t1>>t2>>t3)
    {
        vector<double> tt;
        tt.push_back(t1);
        tt.push_back(t2);
        XX.push_back(tt);
        YY.push_back(t3);
    }
    fin2.close();
    cout<<"EAE"<<endl;

        for(int i=0;i<XX.size();i++)
        {
            for(int j=0;j<XX.size();j++)
            {
                K[i][j]=kernel(XX[i],XX[j]);
            }
        }

    double alpha[XX.size()];
    double C=1.0;

    double b=0;
    SMO(XX,YY,C,alpha,b);
    //cout<<b<<endl;
    int correct=0;
    for(int i=0;i<XX.size();i++)
    {
     //   cout<<"judege:"<<XX[i][0]<<' '<<XX[i][1]<<' '<<G(XX[i],YY,alpha,b,XX)<<' '<<YY[i]<<endl;
        if(G(XX[i],YY,alpha,b,XX)*YY[i]>0) correct++;
    }
    cout<<correct<<endl;

    ofstream fout("E:\\wrs.txt");
    double w[2];
    memset(w,0,sizeof w);
    for(int i=0;i<XX.size();i++)
    {
        for(int j=0;j<XX.size();j++)
            w[j]+=alpha[i]*YY[i]*XX[i][j];
    }
    for(int i=0;i<XX[0].size();i++)
    {
        fout<<w[i]<<' ';
    }
    fout<<b<<endl;
    return 0;
}

int main()
{

        read();
        for(int i=0; i<TrainDataSet.size(); i++)
        {
            if(XX.size()>800) break;
            if(TrainLabelSet[i][0]==3)
            {
                XX.push_back(TrainDataSet[i]);
                YY.push_back(1.0);
            }
            if(TrainLabelSet[i][0]==1)
            {
                XX.push_back(TrainDataSet[i]);
                YY.push_back(-1.0);
            }
        }

        for(int i=0;i<XX.size();i++)
        {
            for(int j=0;j<XX.size();j++)
            {
                K[i][j]=kernel(XX[i],XX[j]);
            }
        }



 srand(time(NULL));
   /* double t1,t2,t3;

    for(int i=0;i<1000;i++)
    {
        t1=rand()%100/10.0;
        t2=rand()%100/10.0;
        int p=rand()%2;
        if(p==0) t3=1;
        else {t3=-1;t1=-t1;t2=-t2;}
        vector<double> tt;
        tt.push_back(t1);
        tt.push_back(t2);
        XX.push_back(tt);
        YY.push_back(t3);
    }*/
   // fin1.close();
    double alpha[XX.size()];
    double C=10;
    cout<<XX.size()<<' '<<YY.size()<<endl;

    double b=0;
    SMO(XX,YY,C,alpha,b);
    int correct=0;
    int num=0;


    Mat XXt;
    vector<double> YYt;
   // freopen("E://testdata.txt","r",stdin);


        for(int i=0; i<TestDataSet.size(); i++)
        {
            double t3;
            num++;
            if(TestLabelSet[i][0]==3)
            {
                t3=1;
            }
            else if(TestLabelSet[i][0]==1)
            {
                t3=-1;
            }
            else {num--;continue;}
         //    cout<<"judge"<<G(TestDataSet[i],YY,alpha,b,XX)*t3<<endl;
            if(G(TestDataSet[i],YY,alpha,b,XX)*t3>b) correct++;
        }

    cout<<correct<<' '<<num<<endl;

    return 0;
}
