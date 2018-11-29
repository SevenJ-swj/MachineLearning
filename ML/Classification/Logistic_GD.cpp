///2018-11-28
///Author:Wenjun Shi
///All Rights Reserved

#include <bits/stdc++.h>
#define Mat vector<vector<double> >

using namespace std;

Mat TrainDataSet,TrainLabelSet,TestDataSet,TestLabelSet;
const int BiThreshold=50;
const int ImageSize=28*28;
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
    for(int i=0;i<n;i++)
        res[i]=a[i]*b;
    return res;
}
vector<double> operator - (vector<double> a,vector<double> b)
{
    int n=a.size();
    vector<double> res(n,0);
    for(int i=0;i<n;i++)
        res[i]=a[i]-b[i];
    return res;
}

inline Mat operator + (Mat a,Mat b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[0].size();j++)
        {
            c[i][j]=a[i][j]+b[i][j];
        }
    }
    return c;
}

inline Mat operator - (Mat a,Mat b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[0].size();j++)
        {
            c[i][j]=a[i][j]-b[i][j];
        }
    }
    return c;
}

inline Mat operator * (Mat a,double b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[0].size();j++)
        {
            c[i][j]=a[i][j]*b;
        }
    }
    return c;
}

Mat getEmat(int n,double lamda)
{
    Mat a(n,vector<double>(n,0));
    for(int i=0;i<n;i++){
        a[i][i]=lamda;
    }
    return a;
}

Mat inverse(Mat A)
{
    int n=A.size();
    Mat C(n,vector<double>(n,0));
    for(int i=0;i<n;i++) C[i][i]=1;
    for(int i=0;i<n;i++)
    {
        for(int j=i;j<n;j++)
        {
            if(fabs(A[j][i])>0){
                swap(A[i],A[j]);
                swap(C[i],C[j]);
                break;
            }
        }
        C[i]=C[i]*(1/A[i][i]);
        A[i]=A[i]*(1/A[i][i]);
        for(int j=0;j<n;j++)
        {
            if(j!=i&&fabs(A[j][i])>0){
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
    for(int i=0;i<A.size();i++)
    {
        for(int j=0;j<A[0].size();j++)
        {
            C[j][i]=A[i][j];
        }
    }
    return C;
}

Mat dot(Mat A,Mat B)
{
    Mat C(A.size(),vector<double>(B[0].size(),0));
    for(int i=0;i<A.size();i++)
    {
        for(int j=0;j<B[0].size();j++)
        {
            for(int k=0;k<A[i].size();k++)
            {
                C[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
    return C;
}

Mat getLR_W(Mat X,Mat Y)
{
    return dot(dot(inverse(dot(transport(X),X)),transport(X)),Y);
}

Mat getLRL1_W(Mat X,Mat Y)
{
    const double lamda=0.003,alpha=1;
    Mat W(X[0].size(),vector<double>(1,1));
    double last=1e18;
    double newf=(dot(transport(dot(X,W)-Y),dot(X,W)-Y))[0][0]/(2*X.size());
    double sum=0;
    while( fabs( newf - last) > 1e-6 )
    {   //printMat(W);
        last=newf;
        Mat tmp=dot(transport(X),dot(X,W)-Y)*(1.0/X.size());
        for(int i=0;i<tmp.size();i++)
        {
            if(W[i][0]>0) tmp[i][0]+=alpha;
            else if(W[i][0]<0) tmp[i][0]+=-alpha;
        }
        W = W - tmp*lamda;
        newf=(dot(transport(dot(X,W)-Y),dot(X,W)-Y))[0][0]/(2*X.size());
        //cout<<"newf:"<<newf<<endl;
    }
    return W;
}

// m*n n*1
Mat getSigmoid(Mat x,Mat w)
{
    Mat t=dot(x,w);
    Mat a;
    for(int i=0;i<t.size();i++)
    {
        double sig=1.0/(1+exp(-t[i][0]));
       // cout<<"getSig"<<t[i][0]<< ' '<<sig<<endl;
        vector<double> tmp;
        tmp.push_back(sig);
        a.push_back(tmp);
    }

    return a;
}

double getNewLoss(Mat x,Mat w,Mat y)
{
    Mat tmp=getSigmoid(x,w);
   // cout<<"getSig  "<<y.size();
    double sum=0;
    for(int i=0;i<tmp.size();i++)
    {
        double t1,t2;
        if(tmp[i][0]<1e-6) t1=-1e9;
        else t1=log(tmp[i][0]);
        if(fabs(1-tmp[i][0])<1e-6) t2=-1e9;
        else t2=log(1-tmp[i][0]);
       // cout<<"sum "<<sum<<' '<<y[i][0];//<<' '<<log(tmp[i][0])<<endl;
        sum+=y[i][0]*t1+(1-y[i][0])*t2;
    }
   // cout<<"getSum";
    return sum*(-1.0/tmp.size());
}

Mat getW(Mat X,Mat Y,double step,double precision,double maxIter)
{
    const double lamda=step;
    Mat W(X[0].size(),vector<double>(1,1));
    double last=1e18;
   // cout<<"getNew "<<Y[0][0]<<endl;
    double newf=getNewLoss(X,W,Y);
    //cout<<"WWW"<<newf<<endl;
    double sum=0;
    int cnt=0;
    while( fabs( newf - last) > precision )
    {   //printMat(W);
        cnt++;
        if(cnt>maxIter) break;
        cout<<fixed<<setprecision(6)<<newf<<endl;
        last=newf;
        Mat tmp=dot(transport(X),getSigmoid(X,W)-Y)*(1.0/X.size());
        W = W - tmp*lamda;
        newf=getNewLoss(X,W,Y);
        //cout<<"newf:"<<newf<<endl;
    }
    return W;
}

//1*n   n*1
double getH(Mat &x,Mat &w)
{
    Mat t=dot(x,w);
    double ans=t[0][0];
    return 1.0/(1+exp(-ans));
}

Mat getLRL2_W(Mat X,Mat Y)
{
    double lamda=0.1*X.size();
    return dot(dot(inverse(dot(transport(X),X)+getEmat(X[0].size(),lamda)),transport(X)),Y);
}

void FeatureScaling(Mat X)
{
    for(int j=1;j<X[0].size();j++)
    {
        double sum=0,sum2=0;
        for(int i=0;i<X.size();i++)
        {
            sum+=X[i][j];
            sum2+=X[i][j]*X[i][j];
        }
        double mean=sum/X.size();
        double stad=sqrt((sum2-X.size()*mean*mean)/X.size());
        //cout<<stad<<endl;
        for(int i=0;i<X.size();i++)
        {
            X[i][j]=(X[i][j]-mean)/stad;
        }
    }
    cout<<"Feature scaling complete"<<endl;
}
/*
void run_LR(int method)
{
    Mat W;
    if(method==0) cout<<"L0:"<<endl,W=getLR_W();
    else if(method==1) cout<<"L1:"<<endl,W=getLRL1_W();
    else if(method==2) cout<<"L2:"<<endl,W=getLRL2_W();
    printMat(W);
    cout<<"LOSS:"<<(dot(transport(dot(X,W)-Y),dot(X,W)-Y))[0][0]/(2*X.size())<<endl;
    Mat ans=dot(X,W);
    double mean,sump=0,sum=0,var=0;
    for(int i=0;i<ans.size();i++)
    {
        //cout<<ans[i][0]<<endl;
        sum+=Y[i][0];
        sump+=ans[i][0];
        var+=(ans[i][0]-Y[i][0])*(ans[i][0]-Y[i][0]);
    }
    mean=sump/ans.size();
    var=sqrt(var/ans.size());
    cout<<"SAMPLE MEAN:"<<sum/ans.size()<<" RESULT MEAN:"<<mean<<" RMS:"<<var<<endl;
}*/

int main()
{

    read();
    Mat X,Y;
    vector<double> t;

    for(int i=0;i<TrainDataSet.size();i++){
        vector<double> y;
       if(TrainLabelSet[i][0]==1) y.push_back(1.0);
        else if(TrainLabelSet[i][0]==0) y.push_back(0.0);
        else continue;
        X.push_back(TrainDataSet[i]);
        X[X.size()-1].push_back(1.0);
        Y.push_back(y);
       // cout<<"LABEL:"<<TrainLabelSet[i][0]<<endl;
    }
   // cout<<X.size()<<' '<<Y.size()<<endl;
    Mat W=getW(X,Y,15,1e-6,100000);
    for(int i=0;i<W.size();i++)
    {
        cout<<W[i][0]<<' ';
    }
    cout<<endl;
    int correct=0,num=0;
    for(int i=0;i<TestDataSet.size();i++){
        Mat tX,tY;
        vector<double> y;
        if(TestLabelSet[i][0]==1) y.push_back(1.0);
        else if(TestLabelSet[i][0]==0) y.push_back(0.0);
        else continue;
        tX.push_back(TestDataSet[i]);
        tX[0].push_back(1.0);

        double predict=getSigmoid(tX,W)[0][0];
        //cout<<predict<<endl;
        if(predict>0.5 && TestLabelSet[i][0]==1) correct++;
        else if(predict<0.5 && TestLabelSet[i][0]==0) correct++;
        num++;
    }
    cout<<correct<<' '<<num<<endl;
    return 0;
}

int main2()
{
    Mat XX,YY;
    double t1,t2,t3;
    ifstream fin2("E://traindata4.txt");
    while(fin2>>t1>>t2>>t3)
    {
        vector<double> tt,yy;
        tt.push_back(t1);
        tt.push_back(t2);
        tt.push_back(1.0);
        XX.push_back(tt);
        if(t3<0) t3=0;
        yy.push_back(t3);
        YY.push_back(yy);
    }

    fin2.close();
    cout<<"EAE"<<endl;
    Mat W=getW(XX,YY,0.003,1e-6,100000);
    for(int i=0;i<W.size();i++)
    {
        cout<<W[i][0]<<' ';
    }
    cout<<endl;
    int correct=0;
    for(int i=0;i<XX.size();i++)
    {
        Mat tX;
        tX.push_back(XX[i]);
     //   cout<<"judege:"<<XX[i][0]<<' '<<XX[i][1]<<' '<<G(XX[i],YY,alpha,b,XX)<<' '<<YY[i]<<endl;
         double predict=getSigmoid(tX,W)[0][0];
         cout<<predict<<endl;
         if(predict>0.5 && YY[i][0]==1) correct++;
         else if(predict<0.5 && YY[i][0]==0) correct++;
    }
    cout<<correct<<endl;

    ofstream fout("E:\\wrs.txt");
    for(int i=0;i<W.size();i++)
    {
        fout<<W[i][0]<<' ';
    }
    fout<<endl;
    return 0;
}
