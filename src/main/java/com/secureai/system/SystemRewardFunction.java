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
@Override
public double reward(SystemState oldState, SystemAction action, SystemState currentState) {

    // -------------------
    // Reward shaping (AI-based attack detection)
    // -------------------
    AttackType actual = environment.getCurrentAttackType();
    AttackType predicted = action.getPredictedAttack(); // ← you added this field
    int detectionTime = environment.getStep();

    if (actual != AttackType.NONE) {
        // If there is an ongoing attack, use shaped reward
        return RewardShapingUtils.calculateReward(actual, predicted, detectionTime);
    }

    // -------------------
    // Standard reward based on system efficiency
    // -------------------
    Action modelAction = this.environment.getActionSet().getActions().get(action.getActionId());

    if (oldState.equals(currentState) && !action.checkPreconditions(this.environment, modelAction)) {
        return -2; // Penalty for non-executable action or no effect
    }

    return -(Config.TIME_WEIGHT * (modelAction.getExecutionTime() / this.maxExecutionTime) +
            Config.COST_WEIGHT * (modelAction.getExecutionCost() / this.maxExecutionCost));
}



    /*
    public double reward(SystemAction systemAction, boolean runnable) {

        Action action = this.environment.getActionSet().getActions().get(systemAction.getActionId());
        //if (oldState.equals(currentState))

        if(!runnable) {
            if(!DynDQNMain.training)
                System.out.println("Not executable action has been selected: "+systemAction.getActionId());
            return -100; // This is the reward if the policy choose an action that cannot be run or keeps the system in the same state
        }

        //if(this.environment.isDone())
        //    return 10;

        return -(Config.TIME_WEIGHT * (action.getExecutionTime() / this.maxExecutionTime) + Config.COST_WEIGHT * (action.getExecutionCost() / this.maxExecutionCost));

    }*/



}
