# secureai-java Reinforcement Learning Evaluation
SecureAI: Deep Reinforcement Learning for Self-Protection in Non-Stationary Cloud Architectures

![Logo](assets/secureai.png)

Ce projet s'appuie sur une architecture d'apprentissage par renforcement profond (Deep Q-Learning) pour la sécurisation d'environnements virtuels via la détection et la réponse aux attaques. Il intègre un système d'évaluation avancé permettant de charger un modèle pré-entraîné et d'évaluer ses performances sans relancer l'entraînement.

## 📌 Sujet
Amélioration d’un agent DQN pour l’auto-défense d’un système virtuel (VMs ou conteneurs) via l’apprentissage par renforcement.

## 📖 Basé sur
Projet original issu d’un environnement SecureAI simulé avec des topologies YAML, utilisant Deeplearning4J (RL4J).

- Lien GitHub source initiale : [https://github.com/nom-utilisateur/projet-original](#)

## 🎯 Objectif du projet

- Éviter la surcharge de réentraînement en important un modèle existant.
- Ajouter un mode évaluation-only.
- Générer automatiquement les résultats (récompenses par épisode, histogrammes d’actions) pour analyse comparative.
- Permettre la comparaison entre agent baseline et agent intelligent.

## ⚙️ Améliorations apportées

- 📦 Sauvegarde automatique du modèle (`models/dyn-trained-model.zip`)
- 🔄 Chargement automatique du modèle à l’évaluation (`evaluate()` sans entraîner)
- 🧪 Évaluation reproductible avec export `.csv` :
  - `evaluation_rewards.csv`
  - `action_histogram.csv`
- 💾 Suppression du mode `train()` si `training = false`

## 📊 Résultats obtenus

- Récompenses cumulées par épisode enregistrées.
- Distribution des actions prises par l’agent visualisable.
- Agent évite les actions invalides plus efficacement que le baseline.
- Amélioration de la politique dans un contexte sans attaque.


