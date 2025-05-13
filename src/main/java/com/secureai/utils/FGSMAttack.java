package com.secureai.utils;
import com.secureai.utils.AdversarialAttack;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class FGSMAttack implements AdversarialAttack {
    @Override
    public INDArray generate(INDArray observation, double epsilon) {
        INDArray gradSign = Transforms.sign(observation, true);
        return observation.add(gradSign.mul(epsilon));
    }
}

