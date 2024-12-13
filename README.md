# Kulinerin - Indonesian Food Recognition

![TF](https://raw.githubusercontent.com/Kulinerin-Bangkit-Team-2024/Machine-Learning/refs/heads/main/assets/tensorflow.png)

## Project Overview
Kulinerin is a machine learning project designed to recognize traditional Indonesian dishes using deep learning techniques. By leveraging Convolutional Neural Networks (CNN) and transfer learning, this project aims to accurately classify various Indonesian foods from images, making it a useful tool for culinary exploration, food cataloging, and tourism-related applications.

## Dataset
The dataset consists of images of various traditional Indonesian dishes, categorized into their respective food types. Each image is labeled with the corresponding dish name, enabling accurate training and testing of the model.
- [Dataset Link](https://drive.google.com/file/d/1fl_ZmjcxhFbebQQknr2QcZQYcd4GFa2A/view?usp=sharing)

## Features
- **Data Augmentation**: Enhance dataset diversity for improved model generalization.
- **Convolutional Neural Networks (CNN)**: For high-accuracy image classification.
  
## Requirements
To replicate or further develop the project, you will need the following libraries and frameworks:
- TensorFlow
- Matplotlib
- NumPy
- Pillow

## Implementation Steps
1. Research machine learning models suitable for food recognition tasks.
2. Collect and preprocess the dataset of traditional Indonesian dishes.
3. Split the dataset into training (80%) and validation (20%) sets.
4. Build a machine learning model using CNN with MaxPooling2D as the base architecture.
5. Train the model with the preprocessed dataset.
6. Evaluate the model’s performance using validation data.
7. Test the model with unseen images to measure its real-world performance.
8. Deploy the model using FastAPI for easy integration with web or mobile applications.

## Results
The model achieved an impressive test accuracy of **99%** on the dataset. Below are the accuracy and loss graphs during training and validation:

![Accuracy Graph](https://raw.githubusercontent.com/Kulinerin-Bangkit-Team-2024/Machine-Learning/refs/heads/main/assets/train_accuracy.png)

## Future Work
- Expand the dataset to include a wider variety of traditional Indonesian dishes from different regions.
- Experiment with more advanced neural network architectures to improve accuracy further.
- Integrate the model into a user-friendly mobile or web application.
- Add features for multi-label classification (e.g., identifying multiple dishes in a single image).
- Explore real-time food recognition capabilities.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the open-source community for providing tools and resources.
- Special appreciation to the contributors and collaborators who made this project possible.
