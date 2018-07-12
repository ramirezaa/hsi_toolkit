//g++ myFile.cpp -fopenmp -L/usr/local/lib -lnabo -lgomp
#include "iostream"
#include "stdio.h"
#include <sys/time.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <limits>

#define VIENNACL_WITH_CUDA 1
#define VIENNACL_WITH_EIGEN 1

//
#include "nabo/nabo.h"

//
#include <arpaca.hpp>

//GDAL
#include "stdio.h"
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()
#include "ogr_spatialref.h"

//Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>

//ViennaCL headers
#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"

using namespace Nabo;
using namespace Eigen;
using namespace std;
//using namespace viennacl::linalg;
//./lle_cpu dc_river.tiff dc_river dimsToKeep knn
//exec/./lle_gpu ~/IMAGES/salinasA.tiff salinasA 3 1
//./lle_cpu_final salinasA.tiff salinasA 3 1
//gdal_translate salinasA_cpu_lle_F32.tiff -scale -ot Byte test2.tiff

/*
salloc -p gpu --gres=gpu:1
srun exec/./lle_gpu ~/IMAGES/salinasA_F32.tiff salinasA 3 10
exit
*/


int getKvalue(int);
void printSparseMat(SparseMatrix<float>, int);

int main(int argc, char *argv[])
{
	stringstream buf2;
	
	signed int K_val = 0;
	signed int dimsToKeep = 0;
	signed int img_rows = 0;
	signed int img_cols = 0;
	int img_band_count = 0;
	int total_pixels = 0;
	double tol =0;
	
	GDALDataset*  poDataset;
	GDALRasterBand* poBand;
	GDALAllRegister();
	string IN_IMG = argv[1];
	string OUTNAME = argv[2];
	
	cout << "\n working on image: " << OUTNAME << endl;	
	
	poDataset = (GDALDataset*) GDALOpen(IN_IMG.c_str(), GA_ReadOnly);
		
	img_band_count = poDataset->GetRasterCount();
	img_cols = poDataset->GetRasterYSize();
	img_rows = poDataset->GetRasterXSize();
	total_pixels = img_cols * img_rows;
	K_val = atoi(argv[4]);
	//
	MatrixXf dataset(img_band_count,total_pixels);//initial data matrix
	MatrixXi indices(K_val,total_pixels);//holds index of k-nearest neigbors
	MatrixXf dists2(K_val, total_pixels);	
	NNSearchF* nns;
	
	float* read_imgScan = (float *) CPLMalloc(sizeof(float)*(total_pixels));
	
	printf("\n Cols: %i, Rows: %i, Bands: %i\n", img_cols, img_rows, img_band_count);
	//printf("\n\nStarting to create dataset matrix\n");
	
	/**
	* Create and fill dense matrices from the Eigen library:
	**/

	/***Start timing***/
	struct timeval t1, t2;
	gettimeofday(&t1, 0);	
	/***/
	
	for(int band_index = 1; band_index <= img_band_count; band_index++){

		poBand = poDataset->GetRasterBand(band_index);
		poBand->RasterIO( GF_Read, 0, 0, img_rows, img_cols, read_imgScan, img_rows, img_cols, GDT_Float32, 0, 0 );
		
		for(int col =0; col < total_pixels; col++){
			dataset(band_index-1, col) = *(read_imgScan+col);
		}
	}	
	
	GDALClose((GDALDatasetH)poDataset);
	CPLFree(read_imgScan);

	dimsToKeep = atoi(argv[3]);
	
	float my_reg = 0.00003;//atof(argv[5]);	
		
	if(K_val > img_band_count){ 
		tol = 0.001; //regularlizer in case constrained fits are ill conditioned
		cout << "\n[note: K values > dimensions; regularization will be used]\n" << endl;
	}
	else
		tol=0;
	
	cout << "\n K-Value: " << K_val << endl;
	cout << "\n Dimensions to keep: " << dimsToKeep << endl;

   	//STEP 1: FIND KNN NEIGHBORS
	cout << "\n finding knn\n\n";
	nns = NNSearchF::createKDTreeLinearHeap(dataset);
	nns->knn(dataset, indices, dists2, K_val, 0.8, NNSearchF::SORT_RESULTS);	
	delete nns;// clean up tree
	cout << "\n Solving for reconstructions weights\n";
	
	JacobiSVD<MatrixXf> svd;	
	SparseMatrix<float> W(total_pixels,total_pixels);
	VectorXf b; b.setOnes(K_val);

	dataset.transposeInPlace();
	indices.transposeInPlace();
	
	for(int ii=0; ii<total_pixels; ii++){

		RowVectorXf queryPoint = dataset.row(ii);
		MatrixXf Z(K_val, img_band_count);
		VectorXi xIndices = indices.row(ii);
		
		for(int k = 0; k < K_val; k++)
			Z.row(k) = dataset.row(xIndices(k)) - queryPoint;
		
		MatrixXf C = Z * Z.transpose();
		
		if(K_val > img_band_count){
		    C = C + ( 0.0003*C.trace()*MatrixXf::Identity(K_val,K_val));
		}
        else{
            C = C + (MatrixXf::Identity(C.rows(),C.cols() )*0.001*C.trace());//regularization for images
        }
		svd.compute(C, ComputeThinU | ComputeThinV);

		VectorXf w = svd.solve(b);
		w = w / w.sum();
		
		for( int k = 0; k < K_val; k++ )
			W.coeffRef(ii, xIndices(k) ) = w(k);	
	}
		
	cout << "\n Computting Embedding coordinates \n";
	
	SparseMatrix<float> I(total_pixels, total_pixels);
	
	typedef Triplet<float> Trip;
	vector<Trip> trp;
	
	for(int i=0; i<total_pixels; i++) trp.push_back(Trip(i,i,1));
	I.setFromTriplets(trp.begin(), trp.end());

	W.makeCompressed();
	I.makeCompressed();
	
	SparseMatrix<float> cpu_M = ( (I-W ).transpose() ) * (I-W);
	SparseMatrix<float> cpu_sparseSub = I-W;	
	
	MatrixXf cpu_denseSub = MatrixXf(cpu_sparseSub);
	
	printf("\nerror 1\n");
	viennacl::matrix<float> gpu_denseSub(cpu_denseSub.rows(), cpu_denseSub.cols());
	viennacl::matrix<float> gpu_denseSub_result(cpu_denseSub.rows(), cpu_denseSub.cols());
	viennacl::matrix<float> gpu_sub_result(cpu_denseSub.rows(), cpu_denseSub.cols());
	printf("\nerror 1-1\n");
	viennacl::copy(cpu_denseSub, gpu_denseSub);
	
	gpu_denseSub_result = viennacl::linalg::prod( trans(gpu_denseSub), gpu_denseSub);
	printf("\nerror 2\n");
	MatrixXf cpu_denseSub_result(gpu_denseSub_result.size1(), gpu_denseSub_result.size2());
	viennacl::copy(gpu_denseSub_result, cpu_denseSub_result);
	
	SparseMatrix<float> cpu_sparseSub_result = cpu_denseSub_result.sparseView();
	
	const int num_eigenvalues = dimsToKeep+1;

	arpaca::SymmetricEigenSolver<float> solver = arpaca::Solve(cpu_sparseSub_result, num_eigenvalues, arpaca::MAGNITUDE_SMALLEST);//MAGNITUDE_SMALLEST 

	const Eigen::MatrixXf& eigenvectors = solver.eigenvectors();

	MatrixXf eigenvectors_final = eigenvectors.rightCols(dimsToKeep).transpose() * sqrt(total_pixels);

	float* scan_line = (float *)CPLMalloc(sizeof(float)*(eigenvectors_final.cols()));
	printf("\n Writing Data\n");	
    /**********************************************************************************************/	
	//starting to write data
	GDALRasterBand* multiBand;
	GDALDataset* outData;
	OGRSpatialReference oSRS;	
	GDALDriver *poDriver;
	GDALAllRegister();
	int bandsKept=1;
	if(img_band_count >6 ){
	    bandsKept = dimsToKeep;
	}else if (img_band_count == 6 ){
	    bandsKept = dimsToKeep; //only for multispectral data
	}
	string out = "/home/aramirez39/IMAGES_OUT/" + OUTNAME + "_gpu_lle_F32.tiff";
	const char *pszFormat = "GTiff"; //"EHdr", "GTiff" , "ENVI"
	const char *pszDstFilename = out.c_str();//will write image as Float32-bit --- "*.tiff"
    char **papszMetadata;
	char **papszOptions = NULL;
	char *pszSRS_WKT = NULL;
	
	oSRS.SetUTM( 11, TRUE );
	oSRS.SetWellKnownGeogCS( "NAD27" ); //"NAD27"
    oSRS.exportToWkt( &pszSRS_WKT );
	
	poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    papszMetadata = poDriver->GetMetadata();

	outData = poDriver->Create( pszDstFilename, img_rows, img_cols, bandsKept, GDT_Float32, papszOptions );
    CPLFree( pszSRS_WKT );
    cout << "\n bandsKept = " << bandsKept << endl;
	//recreating image with new data
	for(int band = 1; band<= bandsKept; band++){
		Map<MatrixXf>( scan_line, 1, eigenvectors_final.cols()) =   eigenvectors_final.row(band-1);
		multiBand = outData->GetRasterBand(band);
		multiBand->RasterIO( GF_Write, 0, 0, img_rows, img_cols, scan_line, img_rows, img_cols, GDT_Float32, 0, 0);
	}
	//Once we're done, properly close the dataset
	CPLFree(scan_line);
	GDALClose( (GDALDatasetH) outData );
	
	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
	printf("\n Time to generate:  %3.1f s \n", time);

	return 0;
}
