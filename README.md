# Driver Drowsiness Detection

Application de détection de somnolence du conducteur basée sur l'état des yeux (EAR - Eye Aspect Ratio), avec alerte sonore en temps réel.

## Motivation

Ce projet a été construit pour développer une compétence centrale dans l'IA appliquée : l'annotation d'images et le contrôle qualité de données pour l'entraînement de modèles de vision par ordinateur. La somnolence au volant étant une cause majeure d'accidents, ce projet reproduit un vrai pipeline de computer vision de bout en bout : collecte de données, annotation manuelle rigoureuse, calibration empirique, et déploiement temps réel.

## Pipeline

1. **Collecte de données** : dataset NTHU Drowsy Driver Detection (images pré-classées drowsy/notdrowsy), dataset MRL Eye Dataset pour le volet ML
2. **Annotation manuelle avec CVAT** : 500 frames annotées avec les labels `eye_open` / `eye_closed`, guidelines de qualité définies en amont (voir `reports/annotation_guidelines.md`)
3. **Calcul de l'EAR** : extraction des landmarks du visage via MediaPipe Face Mesh, calcul géométrique de l'Eye Aspect Ratio
4. **Calibration du seuil** : au lieu d'un seuil arbitraire, le seuil de détection a été calculé empiriquement à partir des annotations CVAT croisées avec l'EAR calculé
5. **Détection temps réel** : flux webcam en direct, alerte déclenchée après fermeture prolongée des yeux (évite les faux positifs sur un simple clignement)
6. **Alerte sonore** : son d'alarme en boucle via pygame.mixer, arrêt instantané dès réouverture des yeux

## Résultats de la calibration

| Métrique | Valeur |
|---|---|
| EAR moyen (yeux ouverts) | 0.2746 |
| EAR moyen (yeux fermés) | 0.1715 |
| Seuil retenu | 0.2231 |
| Frames annotées | 500 (477 avec annotation valide) |
| Répartition | 325 eye_open / 152 eye_closed |

## Structure du projet
driver-drowsiness-detection/
├── data/
│ ├── frames/ # frames NTHU utilisées pour l'annotation CVAT
│ ├── annotations/ # exports CVAT (format YOLO)
│ └── mrl_sample/ # échantillon MRL Eye Dataset (volet ML)
├── src/
│ ├── ear_utils.py # calcul EAR à partir des landmarks
│ ├── calibrate_threshold.py # calibration du seuil via annotations CVAT
│ ├── analyze_annotations.py # statistiques sur les annotations
│ ├── detector.py # pipeline temps réel + alerte
│ └── sample_nthu_for_cvat.py # échantillonnage pour annotation
├── reports/
│ └── annotation_guidelines.md
└── requirements.txt


## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python src/detector.py
```

Appuyer sur `q` pour quitter.

## Limitations

- Dataset d'entraînement limité (500 frames annotées manuellement)
- Détection basée uniquement sur l'état des yeux (pas de bâillement ni d'inclinaison de tête)
- Seuil EAR calibré sur un échantillon spécifique, peut nécessiter un réajustement selon la morphologie du visage ou les conditions d'éclairage

## Auteure

Mariam — Étudiante en Génie Logiciel, ISI Ariana