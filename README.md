cat > README.md << 'EOF'
# Driver Drowsiness Detection

Application de détection de somnolence du conducteur (yeux fermés, bâillements, inclinaison de tête) avec système d'alerte en temps réel.

## Pipeline
1. Extraction de frames depuis vidéos (NTHU / YawDD / captures perso)
2. Annotation via CVAT (bounding boxes yeux/bouche)
3. Calcul EAR/MAR via MediaPipe Face Mesh
4. Calibration des seuils à partir des annotations
5. Détection temps réel + alerte sonore
6. (Optionnel) Classifieur ML entraîné sur features EAR/MAR

## Structure
Voir dossiers `src/`, `data/`, `reports/`.

## Statut
🚧 En cours de développement
EOF