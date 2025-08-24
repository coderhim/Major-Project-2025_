Hello
## Domain Generalization for Medical Image Segmentation

[](https://www.google.com/search?q=https://github.com/coderhim/Major-Project-2025_)

### **Project Overview**

This project presents a novel framework for **domain generalization in medical image segmentation**, specifically addressing the significant performance drop that occurs when a model trained on one data modality (e.g., CT scans) is applied to another (e.g., MRI scans). By employing advanced data augmentation and fusion techniques, this framework enables a single model to perform robustly across different imaging domains without requiring retraining.

-----

### **Key Features**

  * **Domain Shift Mitigation**: Developed a framework designed to overcome domain shifts between CT and MRI modalities.
  * **Nonlinear Data Augmentation**: Utilized a **Thin Plate Spline (TPS) transformation** to perform nonlinear data augmentation, enhancing spatial variability while preserving crucial anatomical structures.
  * **Saliency-Aware Augmentation**: Integrated a **SegFormer-based attention mechanism** to generate attention maps that guide the augmentation process, ensuring that the model focuses on the most salient regions.
  * **High Cross-Modality Performance**: Achieved strong and reliable performance in cross-modmodality segmentation, with Dice scores of **88.82% (CT→MRI)** and **83.07% (MRI→CT)**, significantly outperforming traditional strategies.

-----

### **Getting Started**

#### **Prerequisites**

You will need to have Python and Git installed. Clone the repository using the following command:

```bash
git clone https://github.com/coderhim/Major-Project-2025_.git
cd Major-Project-2025_
```

#### **Installation**

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*(Note: The `requirements.txt` file is assumed to be present in the repository and should contain all necessary libraries like PyTorch, OpenCV, etc.)*

-----

### **Usage**

Instructions on how to run the model will go here. This typically includes how to:

  * Set up your dataset.
  * Run the training script.
  * Run the inference script on new data.
  * View the results.

-----

### **Contributing**

Contributions are welcome\! Please feel free to open an issue or submit a pull request if you have suggestions for improvements or bug fixes.
