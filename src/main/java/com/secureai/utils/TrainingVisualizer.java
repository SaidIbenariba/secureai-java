package com.secureai.utils;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.VertxUIServer;
import java.util.Arrays;
import java.util.List;

@Getter
@Setter
public class TrainingVisualizer {

    private final VertxUIServer uiServer; // 👉 Use VertxUIServer instead of UIServer
    private final InMemoryStatsStorage statsStorage;

    public TrainingVisualizer() {
        uiServer = VertxUIServer.getInstance(9000, false, null); // 👉 Create VertxUIServer on port 9000
        statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        try {
            uiServer.start(); // 👉 Don't forget to start it
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public List<TrainingListener> getListeners(int listenerFrequency) {
        return Arrays.asList(
                new StatsListener(statsStorage, listenerFrequency)
        );
    }

    public void attachToModel(Model model, int listenerFrequency) {
        model.setListeners(getListeners(listenerFrequency));
    }

    public void shutdown() {
        uiServer.detach(statsStorage);
        uiServer.stop();

    }
}
