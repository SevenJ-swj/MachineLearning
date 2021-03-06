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

inline Mat operator + (Mat a,double b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[0].size();j++)
        {
            c[i][j]=a[i][j]+b;
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

inline Mat operator / (Mat a,Mat b)
{
    Mat c(a.size(),vector<double>(a[0].size(),0));
    for(int i=0;i<a.size();i++)
    {
        for(int j=0;j<a[0].size();j++)
        {
            if(fabs(b[i][j])<1e-6) c[i][j]=1e6;
            else c[i][j]=a[i][j]/b[i][j];
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


Mat getAllOne(Mat a,double c)
{
    const int n=a.size();
    const int m=a[0].size();
    Mat t(n,vector<double>(m,c));
    return t;
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

Mat multi(Mat A,Mat B)
{
    Mat C(A.size(),vector<double>(B[0].size(),0));
    for(int i=0;i<A.size();i++)
    {
        for(int j=0;j<A[0].size();j++)
        {
            C[i][j]+=A[i][j]*B[i][j];
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


/// one input layer,one hidden layer,one output layer;

int InputSize,HiddenSize,OutputSize;


double relu(double x)
{
    return max(0.0,x);
}

double sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
}

Mat ReluMat(Mat x)
{
    Mat c=x;
    for(int i=0;i<x.size();i++)
    {
        for(int j=0;j<x[0].size();j++)
        {
            c[i][j]=relu(c[i][j]);
        }
    }
    return c;
}

Mat SigMat(Mat x)
{
    Mat c=x;
    for(int i=0;i<x.size();i++)
    {
        for(int j=0;j<x[0].size();j++)
        {
            c[i][j]=sigmoid(c[i][j]);
        }
    }
    return c;
}

double getLoss(Mat out,Mat YY)
{
   // cout<<"get"<<endl;
   // printMat(out);
    double sum=0;
    for(int i=0;i<out.size();i++)
    {
        for(int j=0;j<out[i].size();j++)
        {
            sum+=(out[i][j]-YY[i][j])*(out[i][j]-YY[i][j]);
        }
    }
  //  cout<<"sum"<<sum<<endl;
    return sum;
}

double getCrossLoss(Mat out,Mat YY)
{
   // cout<<"get"<<endl;
   // printMat(out);
    double sum=0;
    for(int i=0;i<out.size();i++)
    {
        for(int j=0;j<out[i].size();j++)
        {
            if(out[i][j]<1e-6) out[i][j]=1e-6;
            if(out[i][j]+1e-6>=1) out[i][j]=1-(1e-6);
            sum-=YY[i][j]*log(out[i][j])+(1-YY[i][j])*log(1-out[i][j]);
        }
    }
  //  cout<<"sum"<<sum<<endl;
    return sum;
}


Mat PostForwar(Mat &XX,Mat &W1,Mat &W2,double B1,double B2)
{
    Mat Hid=dot(XX,W1)+B1;
    Mat Hido=SigMat(Hid);
    //n*c;
    Mat Out=dot(Hid,W2)+B2;
    Mat Outo=SigMat(Out);
    return Outo;
}

double getSum(Mat x,int row)
{
    double sum=0;
    for(int i=0;i<x[row].size();i++)
        sum+=x[row][i];
    return sum;
}

void TrainSqr(Mat XX,Mat YY,Mat &W1,Mat &W2,double &B1,double&B2,double learnRate,double reg)
{
    //n*d;
    double iter=0;
    cout<<"TEST"<<endl;
    Mat Hid=dot(XX,W1)+B1;
    Mat Hido=SigMat(Hid);
    //n*c;
    Mat Out=dot(Hid,W2)+B2;
    Mat Outo=SigMat(Out);
    //printMat(Outo);
    int step=15000;
    double last=1e9;
    double E=getLoss(Out,YY);
    cout<<E<<endl;
     cout<<"TEST"<<endl;
    while(step--)
    {
        iter++;
        //if(fabs(E-last)<1e-6) break;
        Mat dOuto=Outo-YY;
        Mat dOut=multi(dOuto,multi(SigMat(Out),getAllOne(Out,1.0)-SigMat(Out)));
        Mat dW2=dot(transport(Hido),dOut);
        double dB2=getSum(dOut,0);
 //cout<<"TEST2"<<endl;
        Mat dHid=dot(dOut,transport(W2));
        Mat dSig=multi(dHid,multi(SigMat(Hid),getAllOne(Hid,1.0)-SigMat(Hid)));
        Mat dW1=dot(transport(XX),dSig);
        double dB1=getSum(dSig,0);
 //cout<<"TEST3"<<endl;
 //cout<<W1.size()<<' '<<W1[0].size()<<' '<<dW1.size()<<' '<<dW1[0].size()<<endl;
 //cout<<W2.size()<<' '<<W2[0].size()<<' '<<dW2.size()<<' '<<dW2[0].size()<<endl;
        ///regular object
        dW1=dW1+W1*reg;
        dW2=dW2+W2*reg;
        W1=W1-dW1*learnRate;
        W2=W2-dW2*learnRate;
        B1=B1-dB1*learnRate;
        B2=B2-dB2*learnRate;
            //n*d;
        Hid=dot(XX,W1)+B1;
        Hido=SigMat(Hid);
    //n*c;
        Out=dot(Hid,W2)+B2;
        Outo=SigMat(Out);
        //printMat(Outo);
        last=E;
        E=getLoss(Outo,YY);
        cout<<fixed<<setprecision(6)<<"iter:"<<iter<<" Loss:"<<E<<endl;
    }
}

void TrainCrossEntrypy(Mat XX,Mat YY,Mat &W1,Mat &W2,double &B1,double&B2,double learnRate,double reg)
{
      //n*d;
    double iter=0;
    cout<<"TEST"<<endl;
    Mat Hid=dot(XX,W1)+B1;
    Mat Hido=SigMat(Hid);
    //n*c;
    Mat Out=dot(Hid,W2)+B2;
    Mat Outo=SigMat(Out);
    //printMat(Outo);
    int step=15000;
    double last=1e9;
    double E=getCrossLoss(Out,YY);
    cout<<E<<endl;
     cout<<"TEST"<<endl;
    while(step--)
    {
        iter++;
        //if(fabs(E-last)<1e-6) break;
        Mat dOuto=(YY+(-1.0))/(Outo+(-1.0))-YY/Outo;
        Mat dOut=multi(dOuto,multi(SigMat(Out),getAllOne(Out,1.0)-SigMat(Out)));
        Mat dW2=dot(transport(Hido),dOut);
        double dB2=getSum(dOut,0);
 //cout<<"TEST2"<<endl;
        Mat dHid=dot(dOut,transport(W2));
        Mat dSig=multi(dHid,multi(SigMat(Hid),getAllOne(Hid,1.0)-SigMat(Hid)));
        Mat dW1=dot(transport(XX),dSig);
        double dB1=getSum(dSig,0);
 //cout<<"TEST3"<<endl;
 //cout<<W1.size()<<' '<<W1[0].size()<<' '<<dW1.size()<<' '<<dW1[0].size()<<endl;
 //cout<<W2.size()<<' '<<W2[0].size()<<' '<<dW2.size()<<' '<<dW2[0].size()<<endl;
        ///regular object
        dW1=dW1+W1*reg;
        dW2=dW2+W2*reg;
        W1=W1-dW1*learnRate;
        W2=W2-dW2*learnRate;
        B1=B1-dB1*learnRate;
        B2=B2-dB2*learnRate;
            //n*d;
        Hid=dot(XX,W1)+B1;
        Hido=SigMat(Hid);
    //n*c;
        Out=dot(Hid,W2)+B2;
        Outo=SigMat(Out);
        //printMat(Outo);
        last=E;
        E=getCrossLoss(Outo,YY);
        cout<<fixed<<setprecision(6)<<"iter:"<<iter<<" Loss:"<<E<<endl;
    }
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
        tt.push_back(1.0);
        XX.push_back(tt);
        if(t3<0){yy.push_back(1.0);yy.push_back(0.0);}
        else{yy.push_back(0.0);yy.push_back(1.0);}
        YY.push_back(yy);
        //break;
    }

    fin2.close();

    InputSize=XX[0].size();
    HiddenSize=4;
    OutputSize=YY[0].size();
    Mat W1(InputSize,vector<double>(HiddenSize,0));
    Mat W2(HiddenSize,vector<double>(OutputSize,0));
    for(int i=0;i<W1.size();i++)
        for(int j=0;j<W1[0].size();j++)
            W1[i][j]=rand()%100/1.0;
    for(int i=0;i<W2.size();i++)
        for(int j=0;j<W2[0].size();j++)
            W2[i][j]=rand()%100/1.0;
    double B1=0,B2=0;
    //Train(XX,YY,W1,W2,B1,B2,0.001,0.1);
    TrainCrossEntrypy(XX,YY,W1,W2,B1,B2,0.001,1);
    cout<<"res:"<<B1<<' '<<B2<<endl;
    int correct=0;
    Mat prob=PostForwar(XX,W1,W2,B1,B2);
    //printMat(prob);
    for(int i=0;i<prob.size();i++)
    {
        int id=-1;
        double maxP=-1;
        for(int j=0;j<prob[i].size();j++)
        {
            if(prob[i][j]>maxP)
            {
                id=j;
                maxP=prob[i][j];
            }
        }
        //cout<<YY[i][id]<<endl;
        if(YY[i][id]==1) correct++;
    }
    cout<<correct<<"%"<<endl;
    return 0;
}


