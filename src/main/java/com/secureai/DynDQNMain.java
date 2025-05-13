package com.secureai;

import com.secureai.system.SystemAction;
import com.secureai.utils.TrainingVisualizer;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.topology.Topology;
import com.secureai.nn.DynNNBuilder;
import com.secureai.nn.NNBuilder;
import com.secureai.rl.abs.ParallelDQN;
import com.secureai.rl.abs.SparkDQN;
import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import com.secureai.utils.*;
import lombok.SneakyThrows;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;
import org.deeplearning4j.ui.stats.StatsListener;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;


import java.io.IOException;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

public class DynDQNMain {

    public static final BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();
    static QLearningDiscreteDense<SystemState> dql = null;
    static MultiLayerNetwork nn = null;
    static SystemEnvironment mdp = null;
    static Map<String, String> argsMap;

//    public static Integer iteration = 0; // iteration counter
    public static AtomicInteger iteration = new AtomicInteger(0);
    public static boolean evaluate = true; // if true perform evaluation at the end of each training
    public static boolean transferLearning = false; // if true new NN will be initialized from previous one
    public static int maxIterations; // Total number of test iterations
    public static boolean training = true; // true if the process is currently during training (used for console output purposes)


    public static boolean random = false;

    public static void main(String... args) throws InterruptedException {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        BasicConfigurator.configure();
        TimeUtils.setupStartMillis();

        argsMap = ArgsUtils.toMap(args);

        // Test configuration ---------
        evaluate = true;
        transferLearning = false;
        maxIterations = 1;

        runWithThreshold();
        //runWithTimer();
        //-----------------------------


//        TimeUnit.SECONDS.sleep(3); // Dummy way to synchronize threads
        while( iteration.get() < maxIterations ) {
            System.out.println("Iteration " + iteration);
            queue.take().run();
            iteration.incrementAndGet();
        }
        if(evaluate) evaluate();

    }

    public static void runWithThreshold() {
        int EPOCH_THRESHOLD = 200; // After X epochs

        DynDQNMain.setup();

        dql.addListener(new EpochEndListener() {
            @Override
            public ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, StatEntry statEntry) {
                if (iEpochTrainer.getEpochCounter() == EPOCH_THRESHOLD) {
                    System.out.println("THRESHOLD FIRED");
//                  if(evaluate) { evaluate(); }
                    Timer t = new Timer();
                    t.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            stop(() ->{
//                                DynDQNMain.setup();
//                                queue.add(dql::train);
                                if(iteration.get() < maxIterations) runWithThreshold();
                            });
                            t.cancel();
                        }
                    }, 0);
                }
                return null;
            }
        });
        queue.add(DynDQNMain::train);

    }

    public static void runWithTimer() {
        int TIMER_THRESHOLD = 180000; // After 0s and period 15s


        new Timer(true).schedule(new TimerTask() {
            @SneakyThrows
            @Override
            public void run() {
                System.out.println("TIMER FIRED");
                DynDQNMain.stop(() -> {
                    DynDQNMain.setup();
                    queue.add(dql::train);
                });
            }
        }, 0, TIMER_THRESHOLD);
    }


    public static void stop(CallbackUtils.NoArgsCallback callback) {

        if (dql != null) {
            dql.addListener(new TrainingEndListener() {
                @Override
                public void onTrainingEnd() {
                    callback.callback();
                }
            });
            dql.getConfiguration().setMaxStep(0); // 0 default
            dql.getConfiguration().setMaxEpochStep(0); // 0 default
        } else {
            callback.callback();
        }
    }

    public static void setup() {
        // Init The UI server
        TrainingVisualizer visualizer = new TrainingVisualizer();

        //String topologyId = "2-containers";
        //String actionSetId = "2-containers";
        String topologyId = "1-vms";
        String actionSetId = "1-vms";


        System.out.println(String.format("[Dyn] Choosing topology '%s' with action set '%s'", topologyId, actionSetId));

        Topology topology = YAML.parse(String.format("data/topologies/topology-%s.yml", topologyId), Topology.class);
        ActionSet actionSet = YAML.parse(String.format("data/action-sets/action-set-%s.yml", actionSetId), ActionSet.class);


        String x, y;
        switch (iteration.get()){
            case 0: //x = "30000";
                    //topology.getTasks().get("frontend-service").setReplication(2);
                    break;
            case 1: //x = "0";
                    random = true;
                    break;
            case 2: x = "15000";
                break;

            case 3: x = "15000";
                break;
            case 4: x = "15000";
                break;
            default:
                x = "30000";
                break;
        }

        //---------------------------------------------------------------------------------
        // Transfer learning increasing replicas stress test
        /*if(iteration == 0)
            x = "30000";
        else
            x = "15000";

        if(iteration > 0){
            topology.getTasks().get("frontend-service").setReplication(2);
            topology.getTasks().get("cart-service").setReplication(2);
        }
        if(iteration > 1){
            topology.getTasks().get("recomendation-service").setReplication(2);
            topology.getTasks().get("product-catalog-service").setReplication(2);
        }
        if(iteration > 2){
            topology.getTasks().get("checkout-service").setReplication(2);
            topology.getTasks().get("ad-service").setReplication(2);
        }
        if(iteration > 3){
            topology.getTasks().get("email-service").setReplication(2);
            topology.getTasks().get("payment-service").setReplication(2);
        }
        if(iteration > 4){
            topology.getTasks().get("shiping-service").setReplication(2);
            topology.getTasks().get("currency-service").setReplication(2);
        }*/
        //---------------------------------------------------------------------------------

        QLearning.QLConfiguration qlConfiguration = new QLearning.QLConfiguration(
                Integer.parseInt(argsMap.getOrDefault("seed", "42")),                //Random seed
                Integer.parseInt(argsMap.getOrDefault("maxEpochStep", "500")),       //Max step By epoch
                Integer.parseInt(argsMap.getOrDefault("maxStep", "10000")),           //Max step
                Integer.parseInt(argsMap.getOrDefault("expRepMaxSize", "200")),      //Max size of experience replay
                Integer.parseInt(argsMap.getOrDefault("batchSize", "64")),           //size of batches
                Integer.parseInt(argsMap.getOrDefault("targetDqnUpdateFreq", "100")), //target update (hard)
                Integer.parseInt(argsMap.getOrDefault("updateStart", "0")),           //num step noop warmup
                Double.parseDouble(argsMap.getOrDefault("rewardFactor", "0.01")),        //reward scaling
                Double.parseDouble(argsMap.getOrDefault("gamma", "0.99")),            //gamma
                Double.parseDouble(argsMap.getOrDefault("errorClamp", "0.5")),        //td-error clipping
                Float.parseFloat(argsMap.getOrDefault("minEpsilon", "0.01")),         //min epsilon
                Integer.parseInt(argsMap.getOrDefault("epsilonNbStep", "10000")),      //num step for eps greedy anneal
                Boolean.parseBoolean(argsMap.getOrDefault("doubleDQN", "false"))      //double DQN
        );

        System.out.println("Q-Learning configuration: "+qlConfiguration.toString());

        SystemEnvironment newMdp = new SystemEnvironment(topology, actionSet);
        nn = new NNBuilder().build(newMdp.getObservationSpace().size(),
                newMdp.getActionSpace().getSize(),
                Integer.parseInt(argsMap.getOrDefault("layers", "2")),
                Integer.parseInt(argsMap.getOrDefault("hiddenSize", "32")),
                Double.parseDouble(argsMap.getOrDefault("learningRate", "0.0001")));

        // Attach TV to the network
        ScoreIterationListener scoreListener = new ScoreIterationListener(100);
        StatsListener statsListener = new StatsListener(visualizer.getStatsStorage(), /*reportEveryNIterations=*/1);
        nn.setListeners(scoreListener, statsListener);


        if(iteration.get() > 0 && transferLearning){
            nn.setParams(new DynNNBuilder<>((MultiLayerNetwork) dql.getNeuralNet().getNeuralNetworks()[0])
                    .forLayer(0).transferIn(mdp.getObservationSpace().getMap(), newMdp.getObservationSpace().getMap()) //to use Standard Transfer Learning just use replaceIn or replaceOut
                    .forLayer(-1).transferOut(mdp.getActionSpace().getMap(), newMdp.getActionSpace().getMap())
                    .build().params());
        }


        //nn.setMultiLayerNetworkPredictionFilter(input -> mdp.getActionSpace().actionsMask(input));
//        nn.setListeners(new ScoreIterationListener(100));
        //nn.setListeners(new PerformanceListener(1, true, true));
        System.out.println(nn.summary());

        mdp = newMdp;

        String dqnType = argsMap.getOrDefault("dqn", "standard");
        dql = new QLearningDiscreteDense<>(mdp, dqnType.equals("parallel") ? new ParallelDQN<>(nn) : dqnType.equals("spark") ? new SparkDQN<>(nn) : new DQN<>(nn), qlConfiguration);
        try {
            DataManager dataManager = new DataManager(true);
            dql.addListener(new DataManagerTrainingListener(dataManager));
            dql.addListener(new RLStatTrainingListener(dataManager.getInfo().substring(0, dataManager.getInfo().lastIndexOf('/'))));
        } catch (IOException e) {
            e.printStackTrace();
        }


    }


    public static void train(){

        training = true;
        long trainingTime = System.nanoTime();
        dql.train();
        trainingTime = (System.nanoTime() - trainingTime)/1000000000;
        Logger.getAnonymousLogger().info("[Time] Total training time (seconds):"+trainingTime);
        training = false;
    }


    public static void evaluate() {
//        System.out.println("[Play] Starting adversarial evaluation…");
//        int EPISODES = 20;
//        double[] epsilons = new double[]{0.0, 0.01, 0.05, 0.1, 0.2};
//
//        AdversarialEvaluator eval = new AdversarialEvaluator(dql, mdp);
//        AdversarialAttack fgsm    = new FGSMAttack();
//
//        // 1) Clean vs. adversarial reward
//        double cleanAvg = eval.computeCleanReward(EPISODES);
//        double advAvg05 = eval.computeAdversarialReward(fgsm, 0.05, EPISODES);
//        System.out.printf("Clean Reward=%.3f, Adv@0.05 Reward=%.3f%n", cleanAvg, advAvg05);
//
//        // 2) Attack Success Rate
//        double asr05 = eval.computeAttackSuccessRate(fgsm, 0.05, EPISODES);
//        System.out.printf("Attack Success Rate @0.05 = %.2f%%%n", asr05);
//
//        // 3) Greedy Worst-Case Reward
//        double gwc05 = eval.computeGreedyWorstCaseReward(fgsm, 0.05, EPISODES);
//        System.out.printf("Greedy Worst-Case Reward @0.05 = %.3f%n", gwc05);
//
//        // 4) Robustness Curve
//        Map<Double, Double> curve = eval.computeRobustnessCurve(fgsm, epsilons, EPISODES);
//        curve.forEach((eps, r) -> System.out.printf("ε=%.2f → Reward=%.3f%n", eps, r));

//        System.out.println("[Play] Starting experiment [iteration: "+ iteration +"] ");
//        int EPISODES = 10;
//        double rewards = 0;
//        for (int i = 0; i < EPISODES; i++) {
//            mdp.reset();
//            System.out.println("play policy (episode "+(i+1)+")");
//            double reward = dql.getPolicy().play(mdp);
//            rewards += reward;
//            Logger.getAnonymousLogger().info("[Evaluate] Reward (episode "+(i+1)+"): " + reward);
//        }
//        Logger.getAnonymousLogger().info("[Evaluate] Average reward: " + rewards / EPISODES);
//    }
        System.out.println("[Evaluate] Starting experiment [iteration: " + iteration + "]");
        int EPISODES = 10;
        double totalReward = 0;

        for (int ep = 0; ep < EPISODES; ep++) {
            mdp.reset();
            double episodeReward = 0;
            int step = 0;

            while (!mdp.isDone()) {
                SystemState state = mdp.getState();
                INDArray input = Nd4j.create(state.toArray()).reshape(1, state.toArray().length);
                int actionIndex = dql.getPolicy().nextAction(input);
                SystemAction action = mdp.getActionSpace().encode(actionIndex);

                // Simple rule-based attack prediction (for now)
                AttackType predictedAttack = AttackScenarioGenerator.generate();
                if (action.getActionId().toLowerCase().contains("block")) {
                    predictedAttack = AttackType.PORT_SCAN;
                } else if (action.getActionId().toLowerCase().contains("limit")) {
                    predictedAttack = AttackType.DOS;
                }

                action.setPredictedAttack(predictedAttack);
                StepReply<SystemState> reply = mdp.step(actionIndex);

                AttackType actualAttack = mdp.getCurrentAttackType();
                double shapedReward = RewardShapingUtils.calculateReward(actualAttack, predictedAttack, step);

                episodeReward += shapedReward;
                step++;

                Logger.getAnonymousLogger().info(
                        String.format("[Evaluate] Step %d | Actual: %s | Predicted: %s | Reward: %.2f",
                                step, actualAttack, predictedAttack, shapedReward));
            }

            totalReward += episodeReward;
            Logger.getAnonymousLogger().info("[Evaluate] Episode " + (ep + 1) + " Total Reward: " + episodeReward);
        }

        Logger.getAnonymousLogger().info("[Evaluate] Average Reward: " + totalReward / EPISODES);
    }
//    public static double calculateReward() {
//        mdp.get
//    }

}

