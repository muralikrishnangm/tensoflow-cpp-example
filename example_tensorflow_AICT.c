// AAPM TrueCT denoiser model inference
// Input: 5 images of dimension [nx, ny]
// Output: denoised image in the center of size [nx, ny]
// Model: Unet
// By MGM, 05/13/2022

#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main(int argc, char** argv)
{
  if (argc <= 2) { printf("ERROR: ./example.exe models/<model-name> <image>\n"); return -1;}
  // AICT model in: /gpfs/alpine/gen006/proj-shared/irl1/aict_tfkeras/AICT_checkpoint_dir/horovod_4node_chckpt/ckpt-090

  

  // ================================
  // Read model and allocate inputs & outputs
  // ================================

  // Run following (python needed) to figure out model serving tag, signature, names of inputs & outputs: 
  //        saved_model_cli show --dir <path-to-model-dir>
  //        saved_model_cli show --dir <path-to-model-dir> --tag_set serve
  //        saved_model_cli show --dir <path-to-model-dir> --tag_set serve --signature_def serving_default
  // Use the name of 'inputs' and 'outputs'

  //********* Read model
  TF_Graph* Graph = TF_NewGraph();
  TF_Status* Status = TF_NewStatus();

  TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
  TF_Buffer* RunOpts = NULL;

  // Get path to model directory from input
  const char* saved_model_dir = argv[1];
  printf("Model: %s\n", saved_model_dir);
  // model serve tag
  const char* tags = "serve";
  int ntags = 1;

  TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
  if(TF_GetCode(Status) == TF_OK)
  {
    printf("TF_LoadSessionFromSavedModel OK\n");
  }
  else
  {
    printf("%s",TF_Message(Status));
  }

  //****** Get input tensor
  int NumInputs = 1;
  TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

  TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
  if(t0.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
  else
    printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

  Input[0] = t0;

  //********* Get Output tensor
  int NumOutputs = 1;
  TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

  TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
  if(t2.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
  else	
    printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

  Output[0] = t2;

  //*********Print dims and shape of input and output tensors
//   int num_dim_input  = TF_GraphGetTensorNumDims(Graph, *Input,  Status);
//   int num_dim_output = TF_GraphGetTensorNumDims(Graph, *Output, Status);
//   printf("Input  dims: %d\n", num_dim_input);
//   printf("Output dims: %d\n", num_dim_output);
// 
//  TF_GraphGetTensorShape(Graph, *Input, int64_t* dims, int num_dims,Status);


  //********* Allocate data for inputs & outputs
  TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
  TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

  // set the dimensions of the images here
  int ndims = 4;  // TF requires an additional dim: 3+1
  int nx = 32;
  int ny = 32;
  int nz = 5;

  // read images
  //      FILE *imagefile;
  //      float image[nx][ny];
  //      const char* image_dir = argv[2];
  //      imagefile=fopen(image_dir, "r");
  //      for(int i=0; i<nx; i++) {
  //        for (int j=0 ; j<ny; j++) {
  //          fscanf(imagefile,"%lf",&image[i][j]);
  //        }
  //      }
  //      fclose(imagefile);

  FILE *imagefile;
  float images[nz][nx][ny];
  const char* image_dir = argv[2];
  for (int k=0; k<nz; k++) {
    imagefile=fopen(image_dir, "r");
    printf("Loading image: %d\n", k);
    for(int i=0; i<nx; i++) {
      for (int j=0; j<ny; j++) {
        fscanf(imagefile,"%lf",&images[k][i][j]);
      }
    }
    fclose(imagefile);
  }
  printf("Finished reading images!\n");

  // allocate TF arrays
  int64_t dims[] = {1,nx,ny,nz};
  printf("Allocated TF dims\n");
  float   data[nx*ny*nz];
  printf("Allocated TF array\n");
  int ndata = sizeof(float)*nx*ny*nz; // number of bytes not number of elements
  
  int l = 0;
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      for (int j=0; j<ny; j++) {
        data[l] = 1.0; //images[k][i][j];
        l++;
      }
    }
  }

  TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
  if (int_tensor != NULL)
  {
    printf("TF_NewTensor is OK\n");
  }
  else
    printf("ERROR: Failed TF_NewTensor\n");

  InputValues[0] = int_tensor;

  // ================================
  // Run the Session
  // ================================
  TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);

  if(TF_GetCode(Status) == TF_OK)
  {
    printf("Session is OK\n");
  }
  else
  {
    printf("%s",TF_Message(Status));
  }

  // Free memory
  TF_DeleteGraph(Graph);
  TF_DeleteSession(Session, Status);
  TF_DeleteSessionOptions(SessionOpts);
  TF_DeleteStatus(Status);

  // ================================
  // Print outputs
  // ================================
  void* buff = TF_TensorData(OutputValues[0]);
  float* outvalues = (float*)buff;

  float image_out[nx][ny];
  imagefile=fopen("sample_out.txt", "w");
  int k = 0;
  for(int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      image_out[i][j] = outvalues[k];
      printf("%f   ", image_out[i][j]);
      fprintf(imagefile,"%f ",&image_out[i][j]);
      k++;
    }
    fprintf(imagefile,"\n");
    printf("\n");
  }
  fclose(imagefile);

  
  //    printf("Result Tensor :\n");
  //    for(int i=0;i<nx*ny;i++) {
  //      printf("%f\n",outvalues[i]);
  //    }

  return 0;
}
