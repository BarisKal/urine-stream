# Project in Applied Deep Learning
In this PyTorch project we classify images including two forms of water streams which were produced with a prepared plastic piping tip and a plastic bag. The forms of the water streams serve as artificial urine streams. Water streams generated without building up pressure on the plastic bag belong to the **bad** category, whereas water streams generated with pressure belong to the **good** category. Obviously, the problem stated here is a binary classification problem. The idea at the beginning was to create three categories but it wasn’t possible due to limited resources and insufficient water stream preparation kit. Details of the project is explained in [Exercise1-Description.pdf](https://github.com/BarisKal/urine-stream/blob/main/Exercise1-Description.pdf). 
## Dataset
Around 100k images were taken, 44% of category good and 56% of category bad. These images were split in train (60%), validation (20%), and test set (20%). To gain knowledge about constructing networks with different resolutions, the images are provided in 300x300 and 32x32 pixels. Although labels are provided, the prefixes of the image names give the indication to which group the image belongs to:
* Images starting with **g** belong to category good.
* Images starting with **b** belong to category bad.

Furthermore, the images were taken in different environments, outside with no specific (predefined) background and inside with a dark background. The second letter in the image name therefore indicates were the image was taken:

* If the second letter is equal to **d** the image was taken with a dark background.
* If the second letter is equal to **o** the image was taken without a predefined background.
(Around 80% of the images have the dark background because it was much easier to generate the images under these conditions.)
## Running the pipeline
$**python urinestream.py configfile.json**

If needed, change beforehand parameters in configfile.json
## Metrics and plots
The error metric I specified or rather tried to minimize or maximize were training/validation loss and test accuracy. The losses are plotted and can be found [here](https://github.com/BarisKal/urine-stream/tree/main/PyTorch_Project/visualizations/plots).
Due to the nature of the images (mostly with the same dark background), it was possible to obtain a test accuracy of 100% (round at 5 decimal points). Because I'm a beginner, I tried to achieve a test accuracy of 90% but was surprised with the results. Similar results could be also obtained by training the model with images without the dark background. Furthermore, images with 32x32 pixels were sufficient enough to reach this accuracy.

## Model weights and data sets
Can be found inside releases. The model weights are saved under PyTorch_Project -> models -> weights. The imgaes can be placed anywhere but the paths to the labels and images have to be set in the configfile.json.
Images in higher resolution with 300x300 pixels, can be downloaded [here](https://mega.nz/file/bIR2mbKT#ZtVDEW0-N8CzKuxjtt4thHh94hLPZWIRUQmAu8T0B-U)