# Nigerian License Plate Detection and Recognition System (NLPDRS)

An intelligent AI-powered system for detecting and recognizing **Nigerian license plates** using computer vision and deep learning. Built with a custom dataset from Nigerian roads, **YOLOv8-nano** for segmentation, and **PaddleOCR** for character extraction, the system is optimized for real-time deployment in law enforcement, intelligent surveillance, and road safety monitoring.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [API Usage](#api-usage)
- [Sample Use Cases](#sample-use-cases)
- [Performance](#performance)
- [Challenges](#challenges)
- [Recommendations and Future Work](#recommendations-and-future-work)
- [License](#license)

---

## 📖 Introduction

Nigeria’s growing road network faces increasing challenges including:
- Unregistered/stolen vehicles
- Manual vehicle checks
- Road crimes and traffic violations
- Lack of intelligent surveillance

Traditional methods are time-consuming and error-prone. The **NLPDRS** aims to modernize this process through AI-driven **automated license plate detection and recognition**, specifically tailored for **Nigerian plate formats** and environmental conditions.

---

## ✨ Features

- 🎯 **Real-time License Plate Detection** using YOLOv8
- 🔡 **OCR-Based Character Extraction** using PaddleOCR
- 🧠 **Custom Trained on Nigerian Dataset**
- 🕵️ **Police Mode** – Alerts for wanted vehicle plates
- 🛣️ **FRSC Mode** – Retrieves vehicle and owner details
- 📦 Web API powered by FastAPI for scalable integration
- 🔌 Ready for edge deployment on surveillance systems

---

## 🏗️ System Architecture
📷 Image Input
│
▼
🎯 YOLOv8-nano (Plate Detection)
│
▼
🔍 Cropped Plate Image
│
▼
🔡 PaddleOCR (Character Recognition)
│
▼
🧠 Decision Logic (Police / FRSC Mode)
│
▼
📤 Output Result via FastAPI


---

## 🗃️ Dataset

- **Collected by:** EJAZTECH.AI
- **Images:** 300+ Nigerian vehicle images
- **Sources:** Various locations in **Kano State**, e.g., Sabon Gari, Farm Center, Kasuwar Dan Goro, etc.
- **Resolution:** Standardized to `640x640`
- **Annotation Tool:** [CVAT.AI](https://cvat.ai)
- **Annotation Format:** YOLO `.txt`
- **Annotation Method:** **Segmentation-based** (focus on 7-character license number)

---

## 🧠 Model Pipeline

- **📌 Detection Model:** `YOLOv8-nano`
  - Trained specifically on Nigerian license plates
  - Handles distortion, complex lighting, varying angles

- **📌 OCR Engine:** `PaddleOCR`
  - Extracts only the 7-character plate number
  - Works across both old and new plate formats

---

## 🔌 API Usage

### `POST /detect`

Use this endpoint to submit an image and select your detection mode (`Police` or `FRSC`).

**🧾 Form Fields:**

| Field               | Type   | Description                                                   |
|--------------------|--------|---------------------------------------------------------------|
| `file`             | File   | Vehicle image (JPEG/PNG)                                      |
| `mode`             | String | Either `"Police"` or `"FRSC"`                                 |
| `suspected_numbers`| String | *(Police Mode)* Suspected plate numbers (newline-separated)   |
| `frsc_data`        | String | *(FRSC Mode)* Optional JSON: Plate → Vehicle & Owner Mapping  |

**📂 Sample FRSC JSON:**

```json
{
  "ABC123KJ": {
    "owner": "John Doe",
    "vehicle": "Toyota Corolla, 2015, White"
  }
}


```
## 🧪 Sample Use Cases

### 🚓 Law Enforcement Mode (Police)
- Detects **wanted or blacklisted plates**
- Sends **instant alerts** to the user when a match is found

### 🛣️ FRSC Mode
- Retrieves **owner and vehicle details** from the provided database
- Assists in **registration checks** and **ownership validation**

### 🧾 Traffic Analytics
- Logs detected plates for **movement tracking** and **traffic pattern analysis**

### 🛰️ Edge Deployment
- Can be installed on **highway cameras**, **drones**, or **checkpoints**
- Enables **autonomous, real-time monitoring**

---

## 📊 Performance Metrics

Evaluation on held-out 20% test set:

| **Metric**  | **Score** |
|-------------|-----------|
| Precision   | 0.87      |
| Recall      | 0.91      |
| mAP@0.5     | 0.93      |
| mAP@0.90    | 0.74      |

✅ **High performance under diverse Nigerian conditions confirms model reliability and real-world readiness.**

---

## ⚠️ Challenges Faced

- ❌ Bounding box annotations were **inaccurate** → switched to **segmentation**
- ❌ Standard OCR tools **failed on Nigerian formats** → adopted **PaddleOCR**
- 🚫 Difficulty integrating with **CCTV APIs and hardware systems**

---

## 🔭 Recommendations & Next Steps

- 📈 **Expand dataset**: More cities, states, lighting conditions, and plate types
- 🤝 **Partner with FRSC & NPF**: Access official APIs and registration data
- 🚀 **Deploy on edge devices**: Cameras, checkpoints, and mobile drones
- 📴 **Offline version**: Enable use in rural or internet-poor areas

---

## 👨‍💻 Developers

- **🧪 Project By:** EJAZTECH.AI  
- **👨‍🔬 Lead Developer:** Ismail Ismail Tijjani  
- **🏫 Institution:** Bayero University, Kano  
- **📆 Year:** 2025  

---

## 📜 License

This project is licensed under the **MIT License** — free to use, distribute, and build upon.

---

## 🤝 Contributing

We welcome collaboration and improvements!

- 🔱 Fork this repo  
- 🔁 Create a Pull Request  
- ✉️ Or reach out to our team directly  

Your ideas, contributions, and feedback are truly appreciated!

---
