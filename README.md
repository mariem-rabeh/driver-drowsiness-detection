# Driver Drowsiness Detection

Application de detection de somnolence du conducteur (yeux fermes, baillements, inclinaison de tete) avec systeme d'alerte en temps reel.

## Pipeline
1. Extraction de frames depuis videos (NTHU / YawDD / captures perso)
2. Annotation via CVAT (bounding boxes yeux/bouche)
3. Calcul EAR/MAR via MediaPipe Face Mesh
4. Calibration des seuils a partir des annotations
5. Detection temps reel + alerte sonore
6. (Optionnel) Classifieur ML entraine sur features EAR/MAR

## Structure
Voir dossiers src/, data/, reports/.

## Statut
En cours de developpement