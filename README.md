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
DA augments data augmentation as a part of its pipeline, which rescales RGB values into 0.0~1.0 range, randomly rotates up to 50 degrees, and vertically and horizontally shifts by max 20%. 

**3) Use of HDF5 file**
 Using HDF5 file format, it accelerates its learning speed

**4) Activation & Initialization**
ReLU overcomes gradient saturation problem that Sigmoid and Tanh functions contained. ReLU output does not stagnate for positive input values, while it does for negative input values. Leaky-ReLu is improved version of ReLU because it does not die for negative input values since it has slight angle for negative values. Leaky-ReLu function improved DA’s accuracy by 4.51% compared with ReLU. Furthermore, I implemented SELU for DA this week. Unlike ReLU and Leaky-ReLU, SELU does not lose gradient when input is 0, hence it completely overcomes dying ReLU problem. Replacement to SELU just by itself did not improve accuracy compared to Leaky-ReLU, however, the best performance, 88.46%, with the use of SELU, L1 regularization and decaying learning rate at 0.95 %. The following table is the summary of activations and initializers, relevant for my project.

![image](https://user-images.githubusercontent.com/62607343/130621436-9b2ae0da-3ef5-4657-a67c-5fd116c9475d.png)

**5) Drive Signal**
DA signals “Keep going” if there is no pothole and “Stop” if there is any pothole as the following screenshot:

![image](https://user-images.githubusercontent.com/62607343/130621500-8b93766e-eb83-46d2-8c27-d00cc87135ac.png)

