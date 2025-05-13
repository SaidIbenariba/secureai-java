package com.secureai.utils;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class RLStatTrainingListener implements TrainingListener {
    private BufferedWriter writer;
    int lastStep = 0;
    private Stat<Double> stat;

    public RLStatTrainingListener(String path) {
        try {
            writer = new BufferedWriter(new FileWriter(path + "/training-stats.csv"));
            writer.write("step,reward,loss,epsilon\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public ListenerResponse onTrainingStart() {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {
        try {
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer iEpochTrainer) {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, StatEntry statEntry) {
        try {
            int step = (int) statEntry.getStepCounter();
            double reward = statEntry.getReward();
            writer.write(String.format("%d,%.5f\n", step, reward));
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return ListenerResponse.CONTINUE;
    }


    @Override
    public ListenerResponse onTrainingProgress(ILearning iLearning) {
        return ListenerResponse.CONTINUE;
    }
}
