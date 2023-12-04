# Mood Transfer Model
This folder contains an attempt to create a mood transfer model that transforms an input image based on a specified mood. The transformation involves adjusting the image colors to match the provided mood.

## Prerequisites
Before running the code, make sure to download the necessary dataset and model using the following links:
- [Dataset](https://saifmohammad.com/WebDocs/WikiArt-Emotions.zip) (WikiArt emotions dataset by Saif M. Mohammad)
- [Images](https://drive.google.com/drive/folders/1DujMn8_J0jGEflBFwEItTVX0lRLTtExa?usp=sharing)
- [Models](https://drive.google.com/file/d/180HV2lNu-m8PK_LBW-4uX5xXtA4uqWBn/view?usp=drive_link)

1. Create a "dataset" folder and place the downloaded folders inside. 
2. Create an "images" folder and place the images inside. 
3. Place model files in the root folder.

The folder structure should look like this:
```
├── Art_mood
│   ├── Dataset
│   │   ├── WikiArt-Emotions
│   ├── images
│   ├── mood_image_transformer
```

## Files overview
1. **preprocess.ipynb:** This notebook outlines the steps taken to preprocess the images with the emotions dataset. It generates the **df_with_images.csv** and **update_dataframe.csv** files, which contain updated information after preprocessing, including the images' paths.

2. **cgan_attempt.ipynb**: This notebook represents the initial attempt at creating a Conditional Generative Adversarial Network (cGAN) model for mood transfer.

3. **Training.ipynb**: In this notebook, we train the mood transfer model. There are two models discussed: one that utilizes all the emotions in the dataset and another that uses a condensed version of emotions grouped into "happy," "sad," and "neutral." Model 1, with all moods, has an extended training time, making it impractical for use. Model 2, with condensed emotions, is successfully trained. Refer to the notebook for detailed information.

4. **Training.ipynb**: This notebook also includes a demonstration on the use of the trained models - inputting an image to product an output. Note that there are two models available, but they are essentially the same this is because the keywords "happy" and "sad" are embedded into the model for ease of use. When downloading the model, v1 corresponds to "sad", while v2 corresponds to "happy". The first block of code in "model 2" in "testing the outputs" is working.

Feel free to explore each notebook for a more in-depth understanding of the mood transfer model and its training process.