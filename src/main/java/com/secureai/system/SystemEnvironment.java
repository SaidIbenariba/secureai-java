package com.secureai.system;

import com.secureai.Config;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.stateset.State;
import com.secureai.model.topology.Topology;
import com.secureai.rl.abs.SMDP;
import com.secureai.utils.AttackScenarioGenerator;
import com.secureai.utils.AttackType;
import com.secureai.utils.MapCounter;
import com.secureai.utils.Stat;
import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.json.JSONObject;

public class SystemEnvironment implements SMDP<SystemState, Integer, DiscreteSpace> {
    @Getter
    private AttackType currentAttackType;
    @Getter
    private SystemActionSpace actionSpace;
    @Getter
    private ActionSet actionSet;
    @Getter
    private SystemStateSpace observationSpace;
    @Getter
    private SystemState systemState;
    @Getter
    private SystemRewardFunction systemRewardFunction;
    @Getter
    private SystemTerminateFunction systemTerminateFunction;
    @Getter
    private SystemDefinition systemDefinition;

    @Getter
    private int step = 0;
    @Getter
    private int episodes = 0;
    @Getter
    private double cumulativeReward = 0;

    private MapCounter<String> actionCounter;
    private Stat<Double> stat;

    public SystemEnvironment(Topology topology, ActionSet actionSet) {
        this.actionSet = actionSet;
        this.systemDefinition = new SystemDefinition(topology);
        //this.actionSpace = new SystemActionSpace(this, this.actionSet.actionSpaceSize(systemDefinition)); // variable action set
        this.actionSpace = new SystemActionSpace(this); // fixed action set
        this.observationSpace = new SystemStateSpace(this, this.getSystemDefinition().getSystemStateSize());
        this.systemState = new SystemState(this, this.getSystemDefinition().getSystemStateSize());
        this.systemRewardFunction = new SystemRewardFunction(this);
        this.systemTerminateFunction = new SystemTerminateFunction(this);
        this.actionCounter = new MapCounter<>();
        this.stat = new Stat<>("output/mdp", true);
    }

    public void close() {
        this.stat.flush();
    }

    public boolean isDone() {
        return systemTerminateFunction.terminated(this.systemState);
    }

    public SystemState reset() {
        if(this.step != 0){
            this.stat.append(this.cumulativeReward);
            System.out.println(String.format(Config.GREEN+"[Episode %d][Steps: %d][Cumulative Reward: %f]"+Config.RESET+"[Action bins %s]", this.episodes, this.step, this.cumulativeReward, this.actionCounter));
        }

        this.currentAttackType = AttackScenarioGenerator.generate();
        this.systemState.injectAttack("resource1", this.currentAttackType);

        this.systemState.reset();
        this.step = 0;
        this.episodes++;
        this.actionCounter = new MapCounter<>();
        this.cumulativeReward = 0;
        this.stat.flush();
        return this.systemState;
    }

    public StepReply<SystemState> step(Integer a) {
        this.step++;

        SystemState oldState = this.systemState.copy();
        SystemAction action = this.actionSpace.encode(a);
        boolean runnable = action.checkPreconditions(this, this.getActionSet().getActions().get(action.getActionId()));
        action.run(this);
        SystemState currentState = this.systemState.copy();

        /*if(runnable){
            System.out.println(oldState.equals(currentState));
            System.out.println("---------------------------");
            for(int i=0; i<this.getObservationSpace().size(); i++)
            {
                System.out.println(i+") "+this.getObservationSpace().getMap().get(i)+" : "+this.getSystemState().toIntArray()[i]);
            }
            System.out.println("---------------------------");
        }*/

        double reward = systemRewardFunction.reward(oldState, action, currentState);

        boolean done = this.isDone();
        this.actionCounter.increment(String.format("%s-%s", action.getResourceId(), action.getActionId()));
        this.cumulativeReward += reward;

        return new StepReply<>(currentState, reward, isDone(), new JSONObject("{}"));
    }

    public SystemEnvironment newInstance() {
        return new SystemEnvironment(this.systemDefinition.getTopology(), this.actionSet);
    }


    @Override
    public SystemState getState() {
        return this.systemState;
    }

    @Override
    public void setState(SystemState state) {
        this.systemState = state;
    }


    public void printResourceState(String resourceId){
        // Print resource state
        System.out.print(resourceId+":: ");
        for (State s: State.values()) {
            if( this.getSystemState().get(resourceId, s) != null)
                System.out.print(s+":"+this.getSystemState().get(resourceId, s)+"; ");
        }
        System.out.print("\n");
    }

}
