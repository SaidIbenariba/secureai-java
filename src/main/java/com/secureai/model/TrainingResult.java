package com.secureai.model;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

@Setter
@Getter
public class TrainingResult {
    private double averageReward;
    private long totalTrainingTimeSeconds;
    private String timestamp;

    // New fields for richer insights
    private List<Double> episodeRewards; // reward per episode
    private Map<String, Object> configuration; // training config (e.g., gamma, learning rate, etc.)

    public TrainingResult() {}

    public TrainingResult(double averageReward,
                          long totalTrainingTimeSeconds,
                          String timestamp,
                          List<Double> episodeRewards,
                          List<Double> lossHistory,
                          List<Double> epsilonHistory,
                          Map<String, Object> configuration) {
        this.averageReward = averageReward;
        this.totalTrainingTimeSeconds = totalTrainingTimeSeconds;
        this.timestamp = timestamp;
        this.episodeRewards = episodeRewards;
        this.configuration = configuration;
    }
}
