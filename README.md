# Wheat_Disease_Using_TransferLearning

initial commit:- errors with vgg19 implementation, in vgg19 model fc are changed but no additional layers were added, just experimenting with fc layers, not exactly the transfer learning. i will also impliment Xception,  Inceptionv3 and resnet152 in future. 
<br>
second commit- no errors recorded, but failed to allocate memory from CUDA: error type **CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`**
<br>
third commit: CUDA: memory allocation error. perfectly running on CPU. accuracy doesnt seem promising which means changes required in architecture. **This perticular update is performed on jpg and png files only.**
fourth commit: Noe errors, running perfectly on CUDA: but there seems to be memory leakage with the model, after some iterations , getting the error as "error in process 6" vgg model needs to be updated and hyper- parameter tuning required to maintain metrics. additionally softmax is applied on the final layer, not sure corssentropy from pytorch handles softmax by default. 
