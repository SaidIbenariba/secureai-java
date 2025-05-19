package com.secureai.system;

import com.secureai.DQNMain;
import com.secureai.DynDQNMain;
import com.secureai.model.actionset.Action;
import com.secureai.model.actionset.ActionSet;
import com.secureai.model.stateset.State;
import com.secureai.utils.RandomUtils;
import com.secureai.utils.YAML;
import lombok.AllArgsConstructor;
import lombok.Data;
import com.secureai.utils.AttackType;

@Data
@AllArgsConstructor
public class SystemAction {

    private String resourceId;
    private String actionId;
////    private AttackType predictedAttack;
//
//    public SystemAction(String resourceId, String actionId) {
//        this.resourceId = resourceId;
//        this.actionId = actionId;
////        this.predictedAttack = AttackType.NONE;
//    }

    public void run(SystemEnvironment environment) {


        Action action = environment.getActionSet().getActions().get(this.actionId);

        /*if(!DynDQNMain.training){
            environment.printResourceState(resourceId);
            System.out.println("Evaluating Action: "+this.actionId+" : "+this.resourceId);
        }*/

        if (checkPreconditions(environment, action)){

            action.getPostCondition().run(environment.getSystemState(), this.resourceId, RandomUtils.random.nextDouble());
            if(!DynDQNMain.training) {
                System.out.println("RUN Action: " + this.actionId + " -> " + this.resourceId);
                //environment.printResourceState(resourceId);
            }
        }
    }

    public Boolean checkPreconditions(SystemEnvironment environment, Action action) {
        return action.getPreCondition().run(environment.getSystemState(), this.resourceId);
    }

}
