///2018/11/28
///Auhtor:Wenjun Shi
///All Rights Reserved

///
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

void read_label_set(char *path,Mat &LabelSet)
{
    char tmp;
    freopen(path,"rb",stdin);
    for(int i=0;i<8;i++) scanf("%c",&tmp);
    unsigned char lab;
    while(~scanf("%c",&lab)){
        vector<double> t;
        t.push_back((double)lab);
        LabelSet.push_back(t);
    }
}
void printMat(Mat A)
{
    for(int i=0;i<A.size();i++)
    {
        for(int j=0;j<A[i].size();j++)
        {
            cout<<A[i][j]<<' ';
        }
        cout<<endl;
    }
    cout<<"row:"<<A.size()<<' '<<"col:"<<A[0].size()<<endl;
}
void printImage(vector<double> img)
{
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
            if(img[i*28+j]>=50) cout<<"*";
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
    for(int i=0;i<16;i++) scanf("%c",&tmp);
    unsigned char pix;
    while(~scanf("%c",&pix)){
       image.push_back((double)pix);
        if(image.size()==ImageSize)  DataSet.push_back(image),image.clear();
    }
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


Mat getEMat(int n,double lamda)
{
    Mat a(n,vector<double>(n,0));
    for(int i=0;i<n;i++){
        a[i][i]=lamda;
    }
    return a;
}

Mat getEMat(int n,double lamda,double mi)
{
    Mat a(n,vector<double>(n,0));
    for(int i=0;i<n;i++){
        a[i][i]=lamda;
    }
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            a[i][i]-=mi;
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

Mat calM(int t)
{
    vector<double> M[2];
    for(int i=0;i<TrainDataSet[0].size();i++)
    {
        M[0].push_back(0);
        M[1].push_back(0);
    }

    int classCount[2]={0,0};
    for(int i=0;i<TrainDataSet.size();i++)
    {
        int cType=TrainLabelSet[i][0];
        classCount[t==cType]++;
        for(int j=0;j<TrainDataSet[i].size();j++)
            M[t==cType][j]+=TrainDataSet[i][j];
    }
    Mat ans;
    for(int i=0;i<TrainDataSet[0].size();i++)
    {
        M[0][i]/=classCount[0];
        M[1][i]/=classCount[1];
    }
    ans.push_back(M[0]);
    ans.push_back(M[1]);
    return ans;
}

int main2()
{
    Mat x,y;
    srand(time(NULL));
    Mat m0(100,vector<double>(3,0));
    Mat m1(100,vector<double>(3,0));
    for(int i=0;i<100;i++)
    {
        vector<double> t;
        for(int j=0;j<3;j++)
        {
            //0~1;
            t.push_back((rand()%11)/10.0);
            m1[1][0]=1;
        }
        x.push_back(t);
        y.push_back(vector<double>(1,0));
    }

    for(int i=0;i<100;i++)
    {
        vector<double> t;
        for(int j=0;j<3;j++)
        {
            //1~2;
            t.push_back(((rand()%11)+10)/10.0);
        }
        x.push_back(t);
        y.push_back(vector<double>(1,1));
    }
    //x=transport(x);

    return 0;
}

double kernel(vector<double> x,vector<double> y)
{
    double sum=0;
    for(int i=0;i<x.size();i++)
        sum+=x[i]*y[i];
    return sum;
}

Mat Fisher(Mat X,Mat Y)
{
    Mat N,K1,K2,M1,M2;
    Mat C1,C2;
    for(int i=0;i<X.size();i++)
    {
        if(Y[i][0]==-1) C1.push_back(X[i]);
        else C2.push_back(X[i]);
    }
    cout<<"test"<<endl;
    K1.resize(X.size(),vector<double>(C1.size(),0));
    K2.resize(X.size(),vector<double>(C2.size(),0));
    for(int i=0;i<X.size();i++)
    {
        for(int j=0;j<C1.size();j++)
            K1[i][j]=(kernel(X[i],C1[j]));
        for(int j=0;j<C2.size();j++)
             K2[i][j]=(kernel(X[i],C2[j]));
    }
    cout<<K1.size()<<' '<<K1[0].size()<<endl;
    Mat t=dot(K1,(getEMat(K1[0].size(),1.0,(double)C1.size())));
     cout<<"test2"<<endl;
    N=dot(dot(K1,(getEMat(K1[0].size(),1.0,(double)C1.size()))),transport(K1))+
    dot(dot(K2,(getEMat(K2[0].size(),1.0,(double)C2.size()))),transport(K2));
     cout<<"test3"<<endl;
    M1.resize(X.size(),vector<double>(1,0));
    M2.resize(X.size(),vector<double>(1,0));
    for(int j=0;j<X.size();j++)
    {
        for(int k=0;k<C1.size();k++)
            M1[j][0]+=kernel(X[j],C1[k]);
        for(int k=0;k<C2.size();k++)
            M2[j][0]+=kernel(X[j],C2[k]);
    }
     cout<<"test4"<<endl;
    M1=M1*(1.0/C1.size());
    M2=M2*(1.0/C2.size());
    Mat alpha=dot(inverse(N),(M2-M1));
    cout<<"test5"<<endl;
    return alpha;
}

int main()
{

    Mat XX,YY;
    double t1,t2,t3;
    ifstream fin2("E://traindata4.txt");
    while(fin2>>t1>>t2>>t3)
    {
        vector<double> tt,yy;
        tt.push_back(t1);
        tt.push_back(t2);
        XX.push_back(tt);
        yy.push_back(t3);
        YY.push_back(yy);
    }

    fin2.close();
    cout<<"EAE"<<endl;
    Mat alpha=Fisher(XX,YY);
    cout<<"FINISH"<<endl;
    Mat C1,C2;
    for(int i=0;i<XX.size();i++)
    {
        if(YY[i][0]==-1) C1.push_back(XX[i]);
        else C2.push_back(XX[i]);
    }
    cout<<"FINISH"<<endl;
    double midC1=0,midC2=0;
    for(int i=0;i<C1.size();i++)
    {
        for(int k=0;k<XX.size();k++)
        {
            midC1+=alpha[k][0]*kernel(XX[k],C1[i]);
        }
    }
    midC1=midC1/(1.0*C1.size());
    cout<<"FINISH"<<endl;
    for(int i=0;i<C2.size();i++)
    {
        for(int k=0;k<XX.size();k++)
        {
            midC2+=alpha[k][0]*kernel(XX[k],C2[i]);
        }
    }
    midC2=midC2/(1.0*C2.size());
    cout<<midC1<<' '<<midC2<<endl;
    int correct=0;
    for(int i=0;i<XX.size();i++)
    {
        double sum=0;
        for(int k=0;k<XX.size();k++)
        {
            sum+=alpha[k][0]*kernel(XX[k],XX[i]);
        }
        if( (sum-midC1)*(sum-midC1) > (sum-midC2)*(sum-midC2) )
        {
            if(YY[i][0]==1) correct++;
        }
        else if(YY[i][0]==-1) correct++;
    }


    cout<<correct<<endl;

    /*
    ofstream fout("E:\\wrs.txt");
    for(int i=0;i<2;i++)
    {
        fout<<w[i]<<' ';
    }
    fout<<alpha[alpha.size()-1]<<endl;*/
    return 0;
}
