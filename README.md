# Drive Assistant

Drive Assistant (DA) is Python based Deep Learning Model, which classifies road images into two category: with and without potholes.
It deploys a derivative of GoogLeNet, has achieved 88.46% of accuracy of successful detection of potholes in road images without overfitting. 

The big picture of DA’s architecture consists of - Data Import from Kaggle, Pre-processing, CNN, and signals Drive Instruction. 

The pre-processing pipeline uses HDF5 file format, which sped up the process dramatically. The model automatically loads image from my drive, standardize the size and RGB values, augment data, and split them into the training dataset and validation dataset. 

The main body of DA is GoogLeNet structure - convolutional layers, max-pooling layers, inception layers(a bunch of convolutional layers and a max-pooling layer), Global average pooling layer instead of flattening layer, and “sigmoid” function in the output layer. 

In order to improve accuracy, I made the following changes. First, SELU is used as activation function in place of ReLU and Leaky-ReLU, in order to avoid possible gradient saturation and the dying ReLU problem. SELU is applied to inception layers. Also, I changed the numbers of feature map outputs in the first two layers from 64 to 150. Furthermore, I added three fully connected layers (with 1000, 100 and 100 neurons in this sequence) prior to the output layer. 

These additions could potentially have exposed the model to the risk of overfitting. In fact, it did experience slight overfittings when I increased to 175 and 200 feature maps, and deteriorated its result to 82.10% (-4.10%) and 80.64% (-5.56%). Likewise, four additional fully connected layers (with 1000, 1000, 100 and 100 neurons) reduced accuracy to 83.02%(-3.18%). Therefore, the present model maintains 150 feature maps in the first two layers and three fully connected layers as the following diagram shows.
![image](https://user-images.githubusercontent.com/62607343/130621244-2b537868-daa5-4f85-8542-90c3c0e7f688.png)


**1) Data Ingestion**
DA accepts any size of road images without error and standardize it to (276, 368) size.
The following screenshots show the snippets of the data injection pipeline.
![image](https://user-images.githubusercontent.com/62607343/130619846-12cbc509-a5e9-4cb1-a13d-9644dc58bef4.png)

**2) Data Augmentation (Unused)**
DA augments data augmentation as a part of its pipeline, which rescales RGB values into 0.0~1.0 range, randomly rotates up to 50 degrees, and vertically and horizontally shifts by max 20%. ![image](https://user-images.githubusercontent.com/62607343/130620135-23cf3d3e-e0dd-4ad5-a54f-89870d0a710a.png)

**3) Use of HDF5 file**
 Using HDF5 file format, it accelerates its learning speed
![image](https://user-images.githubusercontent.com/62607343/130620029-6bb2db8a-9ac6-48b2-af4f-d46c4064b618.png)

**4) Activation & Initialization**
ReLU overcomes gradient saturation problem that Sigmoid and Tanh functions contained. ReLU output does not stagnate for positive input values, while it does for negative input values. Leaky-ReLu is improved version of ReLU because it does not die for negative input values since it has slight angle for negative values. Leaky-ReLu function improved DA’s accuracy by 4.51% compared with ReLU. Furthermore, I implemented SELU for DA this week. Unlike ReLU and Leaky-ReLU, SELU does not lose gradient when input is 0, hence it completely overcomes dying ReLU problem. Replacement to SELU just by itself did not improve accuracy compared to Leaky-ReLU, however, the best performance, 88.46%, with the use of SELU, L1 regularization and decaying learning rate at 0.95 %. The following table is the summary of activations and initializers, relevant for my project.
![image](https://user-images.githubusercontent.com/62607343/130620404-0a7f6e2c-457d-40f4-8f99-0f2c32a615d8.png)

**5) Drive Signal**
DA signals “Keep going” if there is no pothole and “Stop” if there is any pothole as the following screenshot:
![image](https://user-images.githubusercontent.com/62607343/130619943-6265d902-adce-4589-8916-dc65872a2ae7.png)

