OPENCV_FLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`
LIBS=`gdal-config --libs`
INC_OG=-I/cm/shared/apps/viennacl/1.6.2/include/ -I$(EIGENINC)
INC_MOD=-I/home/aramirez39/myincludes -I$(EIGENINC) -I$(NABO_INCLUDEDIR)
LIBS2=-L$(NABO_LIBRARYDIR) -lnabo -lgomp $(LIBS)
FLAGS=`gdal-config --cflags`
SRC=src
ex=exec
CPPFLAGS = -m64 -I$(HADOOP_HOME)/c++/Linux-amd64-64/include
LARPACK=-L$(LAPACK_DIR) -L$(BLASDIR) -L$(HOME)/ARPACK -llapack -lblas -larpack

#salloc -p gpu --gres=gpu:1
#srun exec/pca_gpu ~/IMAGES/salinasA_F32.tiff salinasA 20 1
#exit
pca_gpu: $(SRC)/pca_gpu.cu                                                                                      
	nvcc -w $(FLAGS) $(INC_MOD) $< -O2 -o  $(ex)/$@ $(LIBS)	

#exec/pca_gpu ~/IMAGES/salinasA.tiff  salinasA_test1 30
pca_cpu: $(SRC)/pca_cpu.cpp
	g++ $(FLAGS) $(INC_MOD) $< -O2 -o $(ex)/$@ $(LIBS)

lle_gpu: $(SRC)/lle_cpu_enhanced_2.cu
	nvcc -w $(FLAGS) $(INC_MOD) $< -O2 -Xcompiler -fopenmp $(LIBS2) -o $(ex)/$@ -L$(LAPACK_DIR) -llapack -L$(BLASDIR) -lblas -L$(HOME)/ARPACK -larpack	
	
lle_cpu: $(SRC)/lle_cpu_enhanced_2.cpp
	g++ -std=c++11 -w $(FLAGS) $(INC_MOD) $< -O2 -fopenmp $(LIBS2) -o $(ex)/$@ -L$(LAPACK_DIR) -llapack -L$(BLASDIR) -lblas -L$(HOME)/ARPACK -larpack
	
	
lle_cpu_old: $(SRC)/lle_cpu.cpp
	g++ -std=c++11 -w $(FLAGS) $(INC_MOD) -I$(NABO_INCLUDEDIR) $< -O2 -fopenmp $(LIBS2) $(LIBS) -o $(ex)/$@ -L$(LAPACK_DIR) -llapack -L$(BLASDIR) -lblas -L$(HOME)/ARPACK -larpack
	
compile:
	nvcc -w $(FLAGS) $(INC_MOD) $(SRC)/lle_cpu_enhanced.cu -O2 -Xcompiler -fopenmp $(LIBS2) -o $(ex)/lle_gpu $(LARPACK)

compile2_old:
	g++ -std=c++11 -w $(FLAGS) $(INC_MOD) $(SRC)/lle_cpu_enhanced.cpp -O2 -fopenmp $(LIBS2) -o $(ex)/lle_cpu $(LARPACK)

nn:
	nvcc -w -I$(EIGENINC) -I$(NABO_INCLUDEDIR) $(SRC)/$@.cu -Xcompiler -fopenmp $(LIBS2) $(LIBS) -o  $(ex)/$@

kmeans:
	g++ -ggdb $(OPENCV_FLAGS) $(SRC)/$@.cpp -o  $(ex)/$@ $(GPU_LIBS) $(OPENCV_LIBS)
	
separate_classes:
	g++ -ggdb $(OPENCV_FLAGS) $(SRC)/$@.cpp -o  $(ex)/$@ $(GPU_LIBS) $(OPENCV_LIBS)

hsiToCsv:
	g++ $(FLAGS) $(SRC)/$@.cpp -o  $(ex)/$@ $(LIBS)

og_kmeans: $(OBJECTS)
	$(CC) -ggdb $(OPENCV_FLAGS) $(SRC)/$@.cpp -o  $(ex)/$@ $(GPU_LIBS) $(OPENCV_LIBS)	

pca_hadoop:
	nvcc -w $(INC_MOD) $(CPPFLAGS) $(SRC)/pca_hadoop.cu -L$(HADOOP_HOME)/c++/Linux-amd64-64/lib -lhadooppipes -lhadooputils -lpthread -g -O2 -o $(ex)/$@

csvToTiff:
	g++ $(FLAGS) $(SRC)/$@.cpp -o  $(ex)/$@ $(LIBS)
	
clean:
	rm  $(ex)/* *.tiff
