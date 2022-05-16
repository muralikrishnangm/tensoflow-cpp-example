// AAPM TrueCT denoiser model inference
// Input: 5 images of dimension [nx, ny]
// Output: denoised image in the center of size [nx, ny]
// Model: Unet
// By MGM, 05/13/2022

#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/c/c_api.h"
#include <vector>
// #include <hdf5.h>


void NoOpDeallocator(void* data, size_t a, void* b) {}

int main(int argc, char** argv)
{
  if (argc <= 1) { printf("ERROR: ./example.exe models/<model-name>\n"); return -1;}
  // AICT model in: /gpfs/alpine/gen006/proj-shared/irl1/aict_tfkeras/AICT_checkpoint_dir/horovod_4node_chckpt/ckpt-090

//   const char* array_name = "../../image_009.h5";
//   auto f_id = H5Fopen( array_name , H5F_ACC_RDONLY , H5P_DEFAULT );
//   H5File file1( FILE_NAME, H5F_ACC_RDONLY );

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
  int64_t dims[] = {1,nx,ny,nz};
  float   data[nx*ny*nz];
  int ndata = sizeof(float)*nx*ny*nz; // number of bytes not number of elements
  
  int l = 0;
  for (int k=0; k<nz; k++) {
    for (int i=0; i<nx; i++) {
      for (int j=0; j<ny; j++) {
        data[l] = 1.0;
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
  float* offsets = (float*)buff;
  printf("Result Tensor :\n");
  for(int i=0;i<nx*ny;i++) {
    printf("%f\n",offsets[i]);
  }

  return 0;
}
