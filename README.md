# 🎥 Video Action Recognition & Security – UCF101

Ce projet implémente un pipeline complet de reconnaissance d’actions humaines dans des vidéos à partir d’un sous-ensemble du dataset **UCF101**.  
Il inclut :
- Préparation des données (extraction de frames),
- Entraînement d’un CNN 2D avec Transfer Learning (MobileNetV2),
- Lissage temporel pour la stabilité des prédictions,
- Suivi des expériences avec MLflow,
- Déploiement via Streamlit et Docker,
- Extension Sécurité : détection **Normal / Tamper**.

---

## 📂 Structure du projet

```text
video-classification-ucf101/
├── app/
│   └── app.py                # Interface Streamlit (démo)
├── data/
│   ├── raw_videos/           # Vidéos UCF101 / tests
│   ├── frames/               # Frames extraites par classe
│   └── tamper/               # Dataset Normal / Tamper
├── models/
│   ├── cnn_action.h5         # Modèle actions
│   ├── cnn_tamper.h5         # Modèle sécurité
│   └── class_indices.json
├── extract_frames.py
├── train_mobilenet.py       # Entraînement modèle actions
├── predict_video.py         # Prédiction + lissage temporel
├── tamper_attack.py         # Génération données tamper
├── train_tamper.py          # Entraînement Normal / Tamper
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
