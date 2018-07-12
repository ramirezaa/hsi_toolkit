//System headers
#include <iostream>
#include <sys/time.h>
#include "stdio.h"
#include <iomanip>
#include <cmath>
#include <vector>

#define VIENNACL_WITH_CUDA 1
#define VIENNACL_WITH_EIGEN 1
//GDAL Headers
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()
#include "ogr_spatialref.h"

/** 
IMPORTANT: Must be set prior to any ViennaCL includes if you want to use 
ViennaCL algorithms on Eigen objects
**/


//Eigen includes
#include<Eigen/Dense>
#include<Eigen/Core>

//ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/lanczos.hpp"
#include "viennacl/io/matrix_market.hpp"
/*
salloc -p gpu --gres=gpu:1
srun exec/./pca_gpu ~/IMAGES/salinasA_F32.tiff salinasA 20 1 1700
srun exec/./pca_gpu ~/IMAGES/oil_pca.tiff oil 5 1 1700
exit
*/
using namespace std;
using namespace Eigen;

void run_tutorial(string IMG, string OUTNAME, int dimsToKeep, int lambda)
{

	GDALDataset*  poDataset;
	GDALRasterBand* poBand;
	GDALAllRegister();
	poDataset = (GDALDataset*) GDALOpen(IMG.c_str(), GA_ReadOnly);
	
	int img_band_count = poDataset->GetRasterCount();
	int img_cols = poDataset->GetRasterYSize();
	int img_rows = poDataset->GetRasterXSize();
	int total_pixels = img_cols * img_rows;
	double* read_imgScan = (double *) CPLMalloc(sizeof(double)*(img_rows*img_cols));
	cout << endl << "working on image: " << OUTNAME << endl;
	
	printf("Cols: %i, Rows: %i, Bands: %i", img_cols, img_rows, img_band_count);
	printf("\n\nStarting to create dataset matrix\n");

	MatrixXd dataset(img_band_count, total_pixels);	
	MatrixXd eigen_vec(img_band_count, img_band_count);
	MatrixXd eigen_cov_mat(img_band_count, img_band_count);  
	MatrixXd transform_mat(img_band_count, img_band_count);
	MatrixXd y_vectors(img_band_count, total_pixels);
	MatrixXd centered_dataset(img_band_count, total_pixels);
	MatrixXd cov_mat(img_band_count, img_band_count);
	
	VectorXd mean_vector(img_band_count);
	VectorXd eigen_val(img_band_count);
	
	
	/**
	* Create and fill dense matrices from the Eigen library:
	**/
	//loading the dataset matrix.
	for(int band_index = 1; band_index <= img_band_count; band_index++){

		poBand = poDataset->GetRasterBand(band_index);
		poBand->RasterIO( GF_Read, 0, 0, img_rows, img_cols, read_imgScan, img_rows, img_cols, GDT_Float64, 0, 0 );
		
		for(int col =0; col < total_pixels; col++){
			dataset(band_index-1, col) = *(read_imgScan+col);
		}
	}
	
	// printf("\nDone creating dataset matrix\n");
	GDALClose((GDALDatasetH)poDataset);
	CPLFree(read_imgScan);
	
	//calculating mean vector
	printf("\ncalculating mean vector\n");
	mean_vector = dataset.rowwise().sum()/total_pixels;

	//initializing covariance matrix
	printf("\ncalculating covariance matrix\n");
	cov_mat.setZero();
	
	//calculating covariance matrix
	cov_mat = dataset * dataset.transpose();
	cov_mat = (cov_mat /total_pixels) - (mean_vector*mean_vector.transpose());
	
	//calculating covariance matrix	
	viennacl::matrix<double> vcl_cov_mat(img_band_count, img_band_count);
	MatrixXd copy_eigen_vec(img_band_count, dimsToKeep);
	viennacl::copy(cov_mat, vcl_cov_mat);

	viennacl::linalg::lanczos_tag ltag(0.75, 
	                                     dimsToKeep, 
                                        viennacl::linalg::lanczos_tag::no_reorthogonalization, 
                                        lambda);
	
	viennacl::matrix<double, viennacl::column_major> approx_eigenvectors_A(vcl_cov_mat.size1(), ltag.num_eigenvalues());
	std::vector<double> lanczos_eigenvalues = viennacl::linalg::eig(vcl_cov_mat, approx_eigenvectors_A, ltag);
	lanczos_eigenvalues.clear();
	cout << "\nRunning Lanczos algorithm for eigenvectors\n";

	viennacl::copy(approx_eigenvectors_A, copy_eigen_vec);

	copy_eigen_vec.transposeInPlace();
	
	MatrixXd centered_data(img_band_count, total_pixels);
	
	centered_data = dataset.colwise() - mean_vector;
	printf("\nMapping vectors\n");
	
	y_vectors = copy_eigen_vec * centered_data;

	double* scan_line = (double *) CPLMalloc(sizeof(double)*(img_rows*img_cols));
	
    printf("\nWriting Data\n");	
    /***************************************************************************************************************************/	
	//starting to write data
	GDALRasterBand* multiBand;
	GDALDataset* outData;
	OGRSpatialReference oSRS;	
	GDALDriver *poDriver;
	GDALAllRegister();
	int bandsKept;
	if(img_band_count >6 ){
	    bandsKept = dimsToKeep;
	}else if (img_band_count == 6 ){
	    bandsKept = dimsToKeep; //only for multispectral data
	}
	string out = "/v1/aramirez39/Desktop/IMAGES_OUT/" + OUTNAME + "_gpu_pca_F32.tiff";
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

	outData = poDriver->Create( pszDstFilename, img_rows, img_cols, bandsKept, GDT_Float64, papszOptions );
    CPLFree( pszSRS_WKT );
    cout << "bandsKept = " << bandsKept << endl;
	//recreating image with new data
	for(int band = 1; band<= bandsKept; band++){
		Map<MatrixXd>( scan_line, 1, y_vectors.cols()) =   y_vectors.row(band-1);
		multiBand = outData->GetRasterBand(band);
		multiBand->RasterIO( GF_Write, 0, 0, img_rows, img_cols, scan_line, img_rows, img_cols, GDT_Float64, 0, 0);

	}
	
	//Once we're done, properly close the dataset
	CPLFree(scan_line);
	GDALClose( (GDALDatasetH) outData );
}

float standard_deviation(vector<float> data, int n)
{
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/n);           
}

/**
*   In the main() routine we only call the worker function defined above with both single and double precision arithmetic.
**/
int main(int argc, char *argv[])
{

	string IMG = argv[1];
	string OUT = argv[2];
	int dims = atoi(argv[3]);
	int rep_times = atoi(argv[4]);
	int lambda = atoi(argv[5]);
	
	string write_name = "/v1/aramirez39/THESIS/thesis-code/" + OUT + ".txt";
	ofstream salinasA_test1;
	salinasA_test1.open (write_name.c_str(), std::ofstream::out | std::ofstream::app);
	
	struct timeval t1, t2;
	double time=0.0;
	double total_time=0.0;
	vector<float> data;
	
	for(int i=0; i<rep_times;i++){
		gettimeofday(&t1, 0);	
	
		run_tutorial(IMG, OUT, dims, lambda);
		
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		printf("Time to generate:  %f sec \n", time);
		data.push_back(time);
		total_time += time;
		time = 0.0;		
	}
	double avg_time = total_time/rep_times;
	salinasA_test1 << "\nAverage Time to PCA on GPU for " << rep_times << " times: " << avg_time << " seconds\n";
	
	cout << "\nAverage Time to PCA on GPU for " << rep_times << " times: " << avg_time << " seconds\n";
	cout << "\nAverage Time to run PCA on GPU for " << rep_times << " times: " << avg_time << " seconds\n";
	
	salinasA_test1 << "\nStandard deviation= " << standard_deviation(data, data.size()) << endl;	
	cout << "Standard deviation= " << standard_deviation(data, data.size()) << endl;
	
	salinasA_test1.close();
	
	return 0;

}

