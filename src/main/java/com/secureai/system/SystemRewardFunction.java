package com.secureai.system;

import com.secureai.Config;
import com.secureai.DynDQNMain;
import com.secureai.model.actionset.Action;
import com.secureai.rl.abs.RewardFunction;
import lombok.Getter;
import com.secureai.utils.*;

import java.util.Arrays;

public class SystemRewardFunction implements RewardFunction<SystemState, SystemAction> {

    private SystemEnvironment environment;

    @Getter
    private double maxExecutionTime;
    @Getter
    private double maxExecutionCost;

    public SystemRewardFunction(SystemEnvironment environment) {
        this.environment = environment;

        this.maxExecutionTime = this.environment.getActionSet().getActions().values().stream().map(Action::getExecutionTime).max(Double::compareTo).orElse(0d);
        this.maxExecutionCost = this.environment.getActionSet().getActions().values().stream().map(Action::getExecutionCost).max(Double::compareTo).orElse(0d);
    }

    @Override
    public double reward(SystemState oldState, SystemAction action, SystemState currentState) {
        Action modelAction = this.environment.getActionSet().getActions().get(action.getActionId());

        if (oldState.equals(currentState) && action.checkPreconditions(this.environment, modelAction) != true) {
            return -2; // bad action
        }

        return -(Config.TIME_WEIGHT * (modelAction.getExecutionTime() / this.maxExecutionTime) +
                Config.COST_WEIGHT * (modelAction.getExecutionCost() / this.maxExecutionCost));
    }



//    @Override
//    public double reward(SystemState oldState, SystemAction systemAction, SystemState currentState) {
//
//        Action action = this.environment.getActionSet().getActions().get(systemAction.getActionId());
//
//        if(oldState.equals(currentState) && systemAction.checkPreconditions(this.environment, action) != true  ) {
//            return -2; // This is the reward if the policy choose an action that cannot be run or keeps the system in the same state
//        }
//
//        return -(Config.TIME_WEIGHT * (action.getExecutionTime() / this.maxExecutionTime) + Config.COST_WEIGHT * (action.getExecutionCost() / this.maxExecutionCost));
//    }
//@Override
//public double reward(SystemState oldState, SystemAction action, SystemState currentState) {
//    // 1. Extract actual action definition from the action set
//    Action modelAction = this.environment.getActionSet().getActions().get(action.getActionId());
//
//    // 2. Penalty if action is not executable (fails preconditions)
//    boolean runnable = action.checkPreconditions(this.environment, modelAction);
//    if (!runnable) {
//        if (!DynDQNMain.training)
//            System.out.println("[Reward Debug] Not executable: " + action.getActionId());
//        return -1.0; // Minor penalty (less harsh than -2)
//    }
//
//    // 3. Penalty if action is runnable but had no effect (state unchanged)
//    if (oldState.equals(currentState)) {
//        if (!DynDQNMain.training)
//            System.out.println("[Reward Debug] Action had no effect: " + action.getActionId());
//        return -0.5; // Slight penalty for inefficiency
//    }
//
//    // 4. If in attack mode (during training with attacks), reward classification accuracy
//    AttackType actual = environment.getCurrentAttackType();
//    AttackType predicted = action.getPredictedAttack();
//    if (actual != AttackType.NONE && predicted != null) {
//        return RewardShapingUtils.calculateReward(actual, predicted, environment.getStep());
//    }
//
//    // 5. Standard reward: less time/cost → better
//    double efficiencyReward = -(Config.TIME_WEIGHT * (modelAction.getExecutionTime() / maxExecutionTime) +
//            Config.COST_WEIGHT * (modelAction.getExecutionCost() / maxExecutionCost));
//    return efficiencyReward;
//}



}
