# Project in Applied Deep Learning
In this PyTorch project we classify images including two forms of water streams which were produced with a prepared plastic piping tip and a plastic bag. The forms of the water streams serve as artificial urine streams. Water streams generated without building up pressure on the plastic bag belong to the **bad** category, whereas water streams generated with pressure belong to the **good** category. Obviously, the problem stated here is a binary classification problem. The idea at the beginning was to create three categories but it wasn’t possible to create due to limited resources and insufficient water stream preparation kit. Details of the project is explained in [Exercise1-Description.pdf] (https://github.com/BarisKal/urine-stream/blob/main/Exercise1-Description.pdf). 
## Dataset
Around 100k images were taken, 44% of category good and 56% of category bad. These images were split in train (60%), validation (20%), and test set (20%). To gain knowledge about constructing networks with different resolutions, the images are provided in 300x300 and 32x32 pixels. Although labels are provided, the prefixes of the image names give the indication to which group the image belongs to:
* Images starting with **g** belong to category good.
* Images starting with **b** belong to category bad.\

Furthermore, the images were taken in different environments, outside with no specific (predefined) background and inside with a dark background. The second letter in the image name therefore indicates were the image was taken:

* If the second letter is equal to **d** the image was taken with a dark background.
* If the second letter is equal to **o** the image was taken without a predefined background.
(Around 80% of the images have the dark background because it was much easier to generate the images under these conditions.)
## Running the pipeline
**python urinestream.py configfile.json**
If needed, change beforehand parameters in configfile.json
# TODO! Provide data sets