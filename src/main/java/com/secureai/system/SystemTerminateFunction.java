package com.secureai.system;

import com.secureai.rl.abs.TerminateFunction;
import com.secureai.model.stateset.State;
import scala.collection.concurrent.Debug;

public class SystemTerminateFunction implements TerminateFunction<SystemState> {

    private SystemEnvironment environment;

    public SystemTerminateFunction(SystemEnvironment environment) {
        this.environment = environment;
    }

    @Override
    public boolean terminated(SystemState systemState) {

        // Model 1 VMs
        Boolean active, corrupted, appAvailable, dockerRuncUpdated, dockerExecAvailable;
        for (String resourceId : this.environment.getSystemDefinition().getResources()){
            active = systemState.get(resourceId, State.active);
            corrupted = systemState.get(resourceId, State.corrupted);
            appAvailable= systemState.get(resourceId, State.appAvailable);
            dockerRuncUpdated = systemState.get(resourceId, State.dockerRuncUpdated);
            dockerExecAvailable = systemState.get(resourceId, State.dockerExecAvailable);
            if (    (active != null && !active ) ||
                    ( appAvailable != null && !appAvailable) ||
                    ( corrupted != null && corrupted) ||
                    ( dockerRuncUpdated != null && dockerExecAvailable != null && !(dockerRuncUpdated || !dockerExecAvailable)))
                return false;
        }
        return true;

        //-------------------------------------------------------------------------------------
/*
        // Model 2 containers
        Boolean active, corrupted, shellCorrupted, cartCorrupted, confidentialityVuln, integrityVuln;
        for (String resourceId : this.environment.getSystemDefinition().getResources()){
            active = systemState.get(resourceId, State.active);
            corrupted = systemState.get(resourceId, State.corrupted);
            shellCorrupted = systemState.get(resourceId, State.shellCorrupted);
            cartCorrupted = systemState.get(resourceId, State.cartCorrupted);
            confidentialityVuln = systemState.get(resourceId, State.confidentialityVulnerability);
            integrityVuln = systemState.get(resourceId, State.integrityVulnerability);

            if (    ( active != null && !active ) ||
                    ( corrupted != null && corrupted) ||
                    ( shellCorrupted != null && shellCorrupted) ||
                    ( cartCorrupted != null && cartCorrupted) ||
                    ( confidentialityVuln != null && confidentialityVuln) ||
                    ( integrityVuln != null && integrityVuln) )
                return false;
        }
        return true;
*/


    }


}
