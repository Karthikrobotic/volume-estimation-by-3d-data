# 🚀 Live Pothole Detection & Segmentation with Machine Learning

## 📌 Project Overview
In this project, we aim to **detect potholes in live video streams** from a camera mounted on a moving truck.  
Rather than just detecting pothole locations, we **segment the exact pothole areas** to estimate their volume accurately — which is crucial for real-world road maintenance and planning.

---

## 🧠 Why Segmentation & Volume?
✅ Simple object detection only gives a bounding box — insufficient to calculate area or volume.  
✅ With **segmentation**, we get the precise shape of each pothole in every frame, allowing us to:
- Measure area & volume.
- Track the same pothole across multiple frames.
- Handle different shapes and depths.

---

## ⚙️ Model Selection Process
To detect potholes accurately and quickly on live video, we evaluated:
- **YOLOv8** (You Only Look Once, version 8)  
- **Faster R-CNN** (Region-based Convolutional Neural Network)

Both models were trained **with segmentation heads** so they don’t just detect, but also provide the mask (segmented shape).

We selected these models based on:
- 🔍 **Accuracy** (mAP score)
- ⚡ **Speed** (frames per second)

---

## 📦 Dataset & Annotation
- Data collected from **Roboflow** and **Kaggle** datasets containing pothole images.
- Annotated using **Roboflow** for segmentation masks. 
- Dataset directly imported into training pipelines from Roboflow for convenience.
- **Dataset Link:** You can access and explore the complete dataset used in this project on Roboflow Universe:[Pothole Segmentation Dataset on Roboflow Universe](https://universe.roboflow.com/car-xztrx/pot-bwaur)

---

## 🏋️ Model Training
Training was done in **Google Colab** using GPU acceleration:
- Trained both YOLOv8 and Faster R-CNN.
- Tuned hyperparameters for better segmentation performance.
- Compared results to select the best model.
- Finally save the model weights after training.

---

## 📊 Results & Analysis
| Model        | mAP (Accuracy) | FPS (Speed) | Comment                                         |
| ------------ | -------------- | ----------- | ----------------------------------------------- |
| YOLOv8       | ✅ High        | ✅ Fast     | Best choice for live detection & segmentation  |
| Faster R-CNN | ✅ High        | ❌ Slower   | Good accuracy, but lower FPS in live video     |

Based on these results, **YOLOv8** was chosen for final deployment because it delivers:
- **High accuracy segmentation**
- **Real-time performance**

---

## 🧩 Key Points That Make This Project Unique
✅ Uses **segmentation**, not just detection — enables pothole volume calculation.  
✅ Designed for **live video on a moving truck** — handles multiple potholes in real-time.  
✅ Benchmarked models before choosing the final one.  
✅ Practical and industry-focused approach: combining deep learning, data annotation, and real-world deployment.

---

## 📈 Future Work
- Deploy as an **edge AI application** on low-power devices.
- Integrate GPS tagging for pothole location mapping.
- Use temporal tracking to measure pothole growth over time.

---

## 📷 Example Outputs
![Image](https://github.com/user-attachments/assets/7af2c41c-2bc3-44b9-ac93-d59426ac0047)
![Image](https://github.com/user-attachments/assets/1a1950fb-7ce5-485b-bb3e-38a8e5ae619c)


---

## 🙌 Final Note
This project shows how AI and computer vision can **transform road inspection**, making it faster, cheaper, and more precise.

If you'd like, I can also help format the screenshots, diagrams, or add a **project architecture diagram** for an even more professional touch!
