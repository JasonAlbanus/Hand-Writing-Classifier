# Hand-Writing-Classifier

Hand writing classifier for Machine Learning class project

## Project Description
This project focuses on handwriting recognition, which means turning handwritten
text in images into digital text. While scanning documents is easy, converting handwriting 
into machine-readable text is hard because OCR (Optical Character Recognition) doesn't work well with handwriting. 
That’s because handwriting varies a lot in shape, size, and style. To solve this, we use a deep learning model. 
We train it using the IAM dataset, which includes pre-segmented words. The images are cleaned, resized, and labeled so the model can learn. 
This repository includes the source code for this project as well as the project report.

### Dataset Used

Dataset link: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## Testing the Model

See the Utilities section to download a pre-trained model hosted on a file
sharing server.

Once the model file is downloaded, the frontend can be testing with the sample
images stored in ./sample_images.

Additional images can be found within the dataset directory (if downloaded) at
./handwriting-dataset/words/\*/\*/\*.png.

To run the GUI pre-trained model prediction file:

`$> python3 ./predict-handwriting.py cnn2`

Note, the original model has not been configured to work with the current
frontend code.

## Utilities

Both downloading scripts will automatically extract the zip to the correct
folder (as needed by the frontend and dataset imports).

### To download the sample data (for model training)

Note, this is a fairly large file, and due to bandwidth limitations, it can take
around 15-20 minutes to fully download.

- `$> python3 ./download-dataset.py`

### To download the pre-trained models

- `$> python3 ./download-pre-trained-models.py`
