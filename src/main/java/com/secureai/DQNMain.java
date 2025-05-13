package com.secureai;

import com.secureai.model.topology.Topology;
import com.secureai.model.TrainingResult;
import com.secureai.model.actionset.ActionSet;
import com.secureai.nn.FilteredMultiLayerNetwork;
import com.secureai.nn.NNBuilder;
import com.secureai.rl.abs.ParallelDQN;
import com.secureai.rl.abs.SparkDQN;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.*;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;
import java.text.SimpleDateFormat;
import java.io.File;
import java.util.Date;
import java.util.ArrayList;
import java.util.HashMap;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.ui.stats.StatsListener;

public class DQNMain {
    public static boolean training = true;

    public static void main(String... args) throws IOException {
        TrainingVisualizer Visualizer = new TrainingVisualizer();

        // Setup
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        BasicConfigurator.configure();
        TimeUtils.setupStartMillis();

        Map<String, String> argsMap = ArgsUtils.toMap(args);

        // Load Topology & Action Set
        Topology topology = YAML.parse(String.format("data/topologies/topology-%s.yml", argsMap.getOrDefault("topology", "1-vms")), Topology.class);
        ActionSet actionSet = YAML.parse(String.format("data/action-sets/action-set-%s.yml", argsMap.getOrDefault("actionSet", "1-vms")), ActionSet.class);

        // Q-Learning Config
        QLearning.QLConfiguration qlConfiguration = new QLearning.QLConfiguration(
                Integer.parseInt(argsMap.getOrDefault("seed", "42")),
                Integer.parseInt(argsMap.getOrDefault("maxEpochStep", "1000")),
                Integer.parseInt(argsMap.getOrDefault("maxStep", "15000")),
                Integer.parseInt(argsMap.getOrDefault("expRepMaxSize", "5000")),
                Integer.parseInt(argsMap.getOrDefault("batchSize", "128")),
                Integer.parseInt(argsMap.getOrDefault("targetDqnUpdateFreq", "500")),
                Integer.parseInt(argsMap.getOrDefault("updateStart", "100")),
                Double.parseDouble(argsMap.getOrDefault("rewardFactor", "1")),
                Double.parseDouble(argsMap.getOrDefault("gamma", "0.75")),
                Double.parseDouble(argsMap.getOrDefault("errorClamp", "0.5")),
                Float.parseFloat(argsMap.getOrDefault("minEpsilon", "0.01")),
                Integer.parseInt(argsMap.getOrDefault("epsilonNbStep", "15000")),
                Boolean.parseBoolean(argsMap.getOrDefault("doubleDQN", "false"))
        );

        // Build Environment and Network
        SystemEnvironment mdp = new SystemEnvironment(topology, actionSet);
        FilteredMultiLayerNetwork nn = new NNBuilder().build(
                mdp.getObservationSpace().size(),
                mdp.getActionSpace().getSize(),
                Integer.parseInt(argsMap.getOrDefault("layers", "3")),
                Integer.parseInt(argsMap.getOrDefault("hiddenSize", "16")),
                Float.parseFloat(argsMap.getOrDefault("learningRate", "0.001"))
        );
//        nn.setListeners(new ScoreIterationListener(100));
        ScoreIterationListener scoreListener = new ScoreIterationListener(100);
        StatsListener statsListener =
                new StatsListener(Visualizer.getStatsStorage(), /*reportEveryNIterations=*/1);
        nn.setListeners(scoreListener, statsListener);
        System.out.println(nn.summary());

        // DQN type
        String dqnType = argsMap.getOrDefault("dqn", "standard");
        QLearningDiscreteDense<SystemState> dql = new QLearningDiscreteDense<>(
                mdp,
                dqnType.equals("parallel") ? new ParallelDQN<>(nn) :
                        dqnType.equals("spark") ? new SparkDQN<>(nn) :
                                new DQN<>(nn),
                qlConfiguration
        );

        // Add training listeners
        DataManager dataManager = new DataManager(true);
        RLStatTrainingListener rlStatListener = new RLStatTrainingListener(dataManager.getInfo().substring(0, dataManager.getInfo().lastIndexOf('/')));
        dql.addListener(new DataManagerTrainingListener(dataManager));
        dql.addListener(rlStatListener);
        // Start Training
        long startTime = System.nanoTime();
        dql.train();
        long endTime = System.nanoTime();
        long trainingTime = (endTime - startTime) / 1_000_000_000;
        Logger.getAnonymousLogger().info("[Time] Total training time (seconds):" + trainingTime);
        training = false;

        // Evaluate Policy
        System.out.println("[Play] Starting adversarial evaluation…");
        int EPISODES = 20;
        double[] epsilons = new double[]{0.0, 0.01, 0.05, 0.1, 0.2};

        AdversarialEvaluator eval = new AdversarialEvaluator(dql, mdp);
        AdversarialAttack fgsm    = new FGSMAttack();

        // 1) Clean vs. adversarial reward
        double cleanAvg = eval.computeCleanReward(EPISODES);
        double advAvg05 = eval.computeAdversarialReward(fgsm, 0.05, EPISODES);
        System.out.printf("Clean Reward=%.3f, Adv@0.05 Reward=%.3f%n", cleanAvg, advAvg05);
//        int EPISODES = 10;
//        double rewards = 0;
//        ArrayList<Double> episodeRewards = new ArrayList<>();
//        for (int i = 0; i < EPISODES; i++) {
//            mdp.reset();
//            System.out.println("[Play] Starting adversarial evaluation…");
//            double reward = dql.getPolicy().play(mdp);
//            episodeRewards.add(reward);
//            rewards += reward;
//            Logger.getAnonymousLogger().info("[Evaluate] Reward: " + reward);
//        }
//        double averageReward = rewards / EPISODES;
//
//        // Save Result
//        String timestamp = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
//
//        HashMap<String, Object> configMap = new HashMap<>();
//        configMap.put("gamma", qlConfiguration.getGamma());
//        configMap.put("learningRate", argsMap.getOrDefault("learningRate", "0.001"));
//        configMap.put("batchSize", qlConfiguration.getBatchSize());
//        configMap.put("maxEpochStep", qlConfiguration.getMaxEpochStep());
//        configMap.put("epsilonNbStep", qlConfiguration.getEpsilonNbStep());
//        configMap.put("minEpsilon", qlConfiguration.getMinEpsilon());
//        configMap.put("doubleDQN", qlConfiguration.isDoubleDQN());
//        configMap.put("seed", qlConfiguration.getSeed());
//
//        TrainingResult result = new TrainingResult();
//        result.setAverageReward(averageReward);
//        result.setTotalTrainingTimeSeconds(trainingTime);
//        result.setTimestamp(timestamp);
//        result.setEpisodeRewards(episodeRewards);
//        result.setConfiguration(configMap);
//
//        ObjectMapper mapper = new ObjectMapper();
//        try {
//            String outputPath = "src/main/results/result-" + timestamp + ".json";
//            mapper.writeValue(new File(outputPath), result);
//            System.out.println("Result saved to " + outputPath);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
}