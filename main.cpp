#include "NumCpp.hpp"

#include <cstdlib>
#include <bits/stdc++.h>

std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename){

    std::vector<std::pair<std::string, std::vector<double>>> result;
    std::ifstream myFile(filename);

    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    std::string line, colname;
    double val;

    if(myFile.good())
    {
        std::getline(myFile, line);

        std::stringstream ss(line);

        while(std::getline(ss, colname, ',')){
            result.push_back({colname, std::vector<double> {}});
        }
    }

    while(std::getline(myFile, line))
    {
        std::stringstream ss(line);
        
        int colIdx = 0;
        
        while(ss >> val){
            
            result.at(colIdx).second.push_back(val);
            
            if(ss.peek() == ',') ss.ignore();
            
            colIdx++;
        }
    }

    myFile.close();

    return result;
}

// Logistic regression class

class LogisticRegression{
    double sigmoid(double x){
        return 1.0/(1.0 + nc::exp(-1.0*x));
    }

    double h(nc::NdArray<double> x, nc::NdArray<double> w, double b){
        double hx = nc::dot(x, w)[0] + b;
        return sigmoid(hx);
    }

    std::pair<nc::NdArray<double>, double> grad(nc::NdArray<double> y, nc::NdArray<double> x, nc::NdArray<double> w, double b){
        nc::NdArray<double> g_w = nc::zeros<double>(w.shape());
        double g_b = 0.0;

        for (int i=0;i<x.shape().rows;i++){
            double hx = h(x(i,x.cSlice()), w, b);
            g_w = g_w + (y[i] - hx)*x(i,x.cSlice());
            g_b = g_b + (y[i] - hx);
        }
        
        return std::make_pair(g_w/((double)x.shape().rows), g_b/((double)x.shape().rows));
    }

    std::pair<nc::NdArray<double>, double> grad_a(nc::NdArray<double> x, nc::NdArray<double> y, int iterations=2000){
        nc::NdArray<double> w = 2.0*nc::random::rand<double>({x.shape().cols,1}).ravel();
        double b = 5.0*nc::random::rand<double>({1,1})[0];

        for (int i=0;i<iterations;i++){
            std::pair<nc::NdArray<double>, double> grd = grad(y, x, w, b);
            w = w + 0.1*grd.first;
            b = b + 0.1*grd.second;
        }
        
        return std::make_pair(w, b);
    }

    nc::NdArray<double> W;
    double B;
    bool trained = false;

    public:
        LogisticRegression(){}

        void fit(nc::NdArray<double> x, nc::NdArray<double> y, int iterations=2000){
            std::pair<nc::NdArray<double>, double> WB = grad_a(x,y);
            W = WB.first;
            B = WB.second;
            trained = true;
        }

        std::pair<nc::NdArray<double>, double> weights(){
            if (!trained){
                throw std::runtime_error("Model has to be trained first!");
            }
            else{
                return std::make_pair(W, B);
            }
        }
};


int main()
{
    std::vector<std::pair<std::string, std::vector<double>>> features = read_csv("Logistic_X_Train.csv");

    //
    std::vector<double> X1_vec = features[0].second;

    //
    std::vector<double> X2_vec = features[1].second;

    //
    std::vector<double> X3_vec = features[2].second;

    std::vector<std::pair<std::string, std::vector<double>>> targets = read_csv("Logistic_Y_Train.csv");

    std::vector<double> Y = targets[0].second;

    nc::NdArray<double> X_1 = X1_vec;
    nc::NdArray<double> X_2 = X2_vec;
    nc::NdArray<double> X_3 = X3_vec;

    nc::NdArray<double> X_tr = nc::vstack({X_1, X_2, X_3}).swapaxes();

    nc::NdArray<double> Y_tr = Y;

    LogisticRegression model;
    model.fit(X_tr,Y_tr,2000);

    std::pair<nc::NdArray<double>, double> WB = model.weights();

    std::cout <<"Weights: "<< WB.first;
    std::cout <<"Bias: "<< WB.second<<std::endl;

    return 0;
}
