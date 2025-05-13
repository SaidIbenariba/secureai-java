package com.secureai.utils;

import com.secureai.system.SystemEnvironment;
import com.secureai.system.SystemState;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import com.secureai.utils.AdversarialAttack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;



/** Utility to evaluate RL policies under clean and adversarial scenarios. */
public class AdversarialEvaluator {

    private final QLearningDiscreteDense<SystemState> learner;
    private final SystemEnvironment mdp;

    public AdversarialEvaluator(QLearningDiscreteDense<SystemState> learner,
                                SystemEnvironment mdp) {
        this.learner = learner;
        this.mdp = mdp;
    }

    /** Average reward over unperturbed episodes. */
    public double computeCleanReward(int episodes) {
        double total = 0;
        for (int i = 0; i < episodes; i++) {
            mdp.reset();
            total += learner.getPolicy().play(mdp);
        }
        return total / episodes;
    }

    /** Average reward when each observation is perturbed by epsilon. */
    public double computeAdversarialReward(AdversarialAttack attack,
                                           double epsilon,
                                           int episodes) {
        double total = 0;
        for (int i = 0; i < episodes; i++) {
            mdp.reset();
            double epReward = 0;
            while (!mdp.isDone()) {
                INDArray obs    = Nd4j.create(mdp.getState().toArray());
                if(obs.rank() == 1) {
                    obs = obs.reshape(1, obs.length());
                }
                INDArray advObs  = attack.generate(obs, epsilon);

                epReward        += mdp.step(learner.getPolicy().nextAction(advObs)).getReward();
            }
            total += epReward;  
        }
        return total / episodes;
    }

    /** % of episodes where adversarial reward < clean reward (ASR). */
    public double computeAttackSuccessRate(AdversarialAttack attack,
                                           double epsilon,
                                           int episodes) {
        int success = 0;
        for (int i = 0; i < episodes; i++) {
            double clean = computeCleanReward(1);
            double adv   = computeAdversarialReward(attack, epsilon, 1);
            if (adv < clean) success++;
        }
        return 100.0 * success / episodes;
    }

    /** Minimum reward achieved under greedy per‐step perturbations (GWC). */
    public double computeGreedyWorstCaseReward(AdversarialAttack attack,
                                               double epsilon,
                                               int episodes) {
        double worst = Double.MAX_VALUE;
        for (int i = 0; i < episodes; i++) {
            mdp.reset();
            double epReward = 0;
            while (!mdp.isDone()) {
                INDArray obs = Nd4j.create(mdp.getState().toArray());
                if(obs.rank() == 1) {
                    obs = obs.reshape(1, obs.length());
                }
                INDArray advObs = attack.generate(obs, epsilon);
                epReward       += mdp.step(learner.getPolicy().nextAction(advObs)).getReward();
            }
            worst = Math.min(worst, epReward);
        }
        return worst;
    }

    /** Builds a map ε → adversarial average reward for multiple strengths. */
    public Map<Double, Double> computeRobustnessCurve(AdversarialAttack attack,
                                                      double[] epsilons,
                                                      int episodes) {
        Map<Double, Double> curve = new LinkedHashMap<>();
        for (double eps : epsilons) {
            curve.put(eps, computeAdversarialReward(attack, eps, episodes));
        }
        return curve;
    }
}
